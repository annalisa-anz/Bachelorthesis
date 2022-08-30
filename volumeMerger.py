"""
 Das Programm dient dazu, aus zwei MRT-Bilddatensätzen (sog. Volumes),
 welche aus unterschiedlichen Perspektiven aufgenommen wurden,
 einen kombinierten Datensatz zu erzeugen.
 Dadurch wird die Auflösung verbessert und das erzeugte Volume kann von allen Seiten korrekt
 betrachtet werden.

 @autor Annalisa Anzioso
"""
import argparse
import os

import SimpleITK as sItk
import numpy as np
import scipy.interpolate


def read_params():
    """
     Liest die Parameter ein, mit denen das Programm aufgerufen wurde.

     Die ersten beiden Parameter geben die koronale und sagittale Ansicht an, welche kombiniert werden sollen.
     Mit "-r", "--resultfile" kann eine Datei angegeben werden, in die das Ergebnis geschrieben wird.
     Falls das Programm keine Registrierung vornehmen soll, kann es mit "-n", "--noregistration" aufgerufen werden.
     In diesem Fall wird nur die Vorverarbeitung für eine anderweitige Registrierung abgeschlossen.
     Wenn bereits eine Registrierung vorgenommen wurde, kann mit "-m", "--matrixfile" eine Datei
     angegeben werden, die die generierte Matrix aus der Registrierung enthält.
     Mit "-i", "--interpolation3d" kann angegeben werden, dass eine 3D-Interpolation durchgeführt werden soll.
     Standardmäßig nutzt das Programm eine 2D-Interpolation.
     Außerdem wird mit "-o", "--outputdir" ein Pfad angegeben, wo die generierten Dateien abgelegt werden.
     Mit "-f", "--fast" kann angegeben werden, dass bei der Interpolation die Nearest Neighbor Methode genutzt werden
     soll.

     :returns: coronal file, sagittal file, result file, no registration, matrix file, interpolation 3d, output
     directory, fast_interpolation
    """

    # Parsen der Befehlszeile in Python-Datentypen
    parser = argparse.ArgumentParser(description='Merge 2 MRT Volumes to one volume with high resolution.')

    # 1. Dateipfad der koronalen Ansicht
    parser.add_argument("coronal_file", help="The filename of the coronal volume used for merging.")
    # 2. Dateipfad der sagittalen Ansicht
    parser.add_argument("sagittal_file", help="The filename of the sagittal volume used for merging.")

    # 3. [optional] Dateipfad für das Ergebnis
    parser.add_argument("-r", "--resultfile", dest="result_file",
                        help="Result file name of the merged volume. Default is MergedVolume.mhd", metavar="FILE",
                        default="MergedVolume.mhd")

    # -n und -m schliessen sich aus, da ohne Registrierung keine Matrix notwendig ist
    group = parser.add_mutually_exclusive_group()
    # 4. [optional] Angabe, dass das Programm nur bis zur Registrierung laufen soll
    group.add_argument("-n", "--noregistration",
                       action="store_true", dest="no_registration", default=False,
                       help="Stop after resampling. "
                            "The user can perform its own registration (e.g. using merit or something else).")
    # 4. [optional] Dateipfad der Matrix einer stattgefundenen Registrierung
    group.add_argument("-m", "--matrixfile", dest="matrix_file",
                       help="Use the file as 4x4 matrix for affine transformation.", metavar="FILE")

    # 5. [optional] Angabe, dass die 3D-Interpolation genutzt werden soll
    parser.add_argument("-i", "--interpolation3d", type=int, action="store", dest="interpolation3d", metavar="VALUE",
                        default=0,
                        help="Use 3D interpolation and divide each axis in number of segments to build interpolation "
                             "cluster.")

    # 6. [optional] Dateipfad für die generierten Dateien
    parser.add_argument("-o", "--outputdir", dest="output_dir",
                        help="Output Directory where files will be generated. The directory must exist.",
                        metavar="DIRECTORY", default="./")

    # 7. [optional] Angabe, ob Fast Interpolation (Nearest Neighbor) verwendet wird
    parser.add_argument("-f", "--fast",
                        action="store_true", dest="fast_interpolation", default=False,
                        help="Use Nearest Neighbor as interpolation method.")

    args = parser.parse_args()

    # Wenn im Parameter -r kein Pfad enthalten ist, wird output_dir und der Defaultname verwendet
    if (os.path.dirname(args.result_file) is None) or (os.path.dirname(args.result_file) == ""):
        res_file = os.path.join(args.output_dir, args.result_file)
    else:
        res_file = args.result_file

    # Die angegebenen Verzeichnisse müssen vorhanden sein
    if (not os.path.isdir(args.output_dir)) or (not os.path.isdir(os.path.dirname(res_file))):
        print("Error: Specified output directory or path for result file does not exist")
        raise RuntimeError

    # result_file muss die Endung ".mhd" haben
    filename, extension = os.path.splitext(res_file)
    if extension.lower() != ".mhd":
        print("Error: The result filename must have the extension .mhd.")
        raise RuntimeError

    return args.coronal_file, args.sagittal_file, res_file, args.no_registration, args.matrix_file, args. \
        interpolation3d, args.output_dir, args.fast_interpolation


def read_and_check_volumes(coronal_file, sagittal_file):
    """
     Versucht das koronale und das sagittale Volume aus den übergebenen Dateien einzulesen.

     :param coronal_file: koronale Ansicht, die fusioniert werden soll
     :param sagittal_file: sagittale Ansicht, die fusioniert werden soll
     :returns: das eingelesene koronale und sagittale Volume
    """

    try:
        print("Reading coronal volume...")
        cor_vol = sItk.ReadImage(coronal_file)
    except RuntimeError:
        print("Error: Could not read coronal volume")
        raise RuntimeError

    try:
        print("Reading sagittal volume...")
        sag_vol = sItk.ReadImage(sagittal_file)
    except RuntimeError:
        print("Error: Could not read sagittal volume")
        raise RuntimeError

    return cor_vol, sag_vol


def resample_volume_to_iso_voxel(volume, spacing, fast_interpolation):
    """
     Berechnet eine neue Auflösung für ein angegebenes Volume, mithilfe eines angegebenen Referenz-spacings,
     damit die Voxelgrößen in x, y und z gleich sind.

     :param volume: Volume, welches auf ISO-Voxelgröße aufgelöst werden soll
     :param spacing: Referenz-spacing, damit die Voxelgrößen in x, y und z gleich sind
     :param fast_interpolation: Wenn True wird für die Interpolation Nearest Neighbor verwendet
     :returns: Das Volume mit der neu ermittelten Auflösung, welches in x, y und z die gleiche Voxelgröße hat
    """

    # Neues Spacing ermitteln (soll für alle Dimensionen gleich werden)
    new_spacing = ()

    for i in range(0, volume.GetDimension()):
        new_spacing += (spacing,)

    return resample_volume_to_new_voxel(volume, new_spacing, fast_interpolation)


def resample_volume_to_new_voxel(volume, new_spacing, fast_interpolation):
    """
     Berechnet eine neue Auflösung für ein angegebenes Volume, mithilfe von angegebenen Referenz-spacings
     für die Voxelgröße in x, y und z.

     :param volume: Volume, welches auf die neue Voxelgröße aufgelöst werden soll
     :param new_spacing: Tupel für das neue Spacing in x, y, z
     :param fast_interpolation: Wenn True, wird für die Interpolation Nearest Neighbor verwendet
     :returns: Das Volume mit der neu ermittelten Auflösung
    """

    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()

    # Neue Größe ermitteln (je nachdem, ob sich das spacing der Dimension ändert, ändert sich die Größe)
    new_size = ()
    for i in range(0, volume.GetDimension()):
        new_size += (round(original_size[i] * original_spacing[i] / new_spacing[i]),)

    # Auflösung des Volumes neu berechnen, auf denen sich verändernden Achsen wird interpoliert (für Registrierung)
    if fast_interpolation:
        return sItk.Resample(image1=volume, size=new_size, transform=sItk.Transform(),
                             interpolator=sItk.sitkNearestNeighbor,
                             outputOrigin=volume.GetOrigin(), outputSpacing=new_spacing,
                             outputDirection=volume.GetDirection(), defaultPixelValue=0,
                             outputPixelType=volume.GetPixelID())
    else:
        return sItk.Resample(image1=volume, size=new_size, transform=sItk.Transform(), interpolator=sItk.sitkBSpline,
                             outputOrigin=volume.GetOrigin(), outputSpacing=new_spacing,
                             outputDirection=volume.GetDirection(), defaultPixelValue=0,
                             outputPixelType=volume.GetPixelID())


def transform_voxel_of_volume(volume, transformed_volume, transform):
    """
     Transformiert die Voxel eines Volumes und schreibt die transformierten Voxel in das angegebene
     transformierte Volume. Die Voxel werden dabei mit der angegebenen Transformation transformiert.

     :param volume: Volume, welches transformiert werden soll
     :param transformed_volume: Volume, in dem die transformierten Pixel gespeichert werden sollen
     :param transform: Transformation, die auf die Pixel angewendet werden soll
     :returns: Das transformierte Volume
    """

    # Größe des zu transformierenden Volumes speichern
    x_size = volume.GetWidth()
    y_size = volume.GetHeight()
    z_size = volume.GetDepth()

    # Volume durchlaufen und Transformation auf die Pixel anwenden
    for z in range(0, z_size):
        for y in range(0, y_size):
            for x in range(0, x_size):
                # Aktuelles Voxel in Weltkoordinaten umwandeln
                physical_point = volume.TransformIndexToPhysicalPoint([x, y, z])
                transformed_point = physical_point

                # Wenn eine Transformation angegeben wurde, diese für den gegebenen Punkt ausführen
                if transform is not None:
                    transformed_point = transform.TransformPoint(physical_point)

                # Pixelposition im neuen Volume ermitteln
                idx = transformed_volume.TransformPhysicalPointToIndex(transformed_point)

                # Prüfen, ob Pixel in das transformierte Volume passt
                if (idx[0] in range(0, transformed_volume.GetWidth())) & \
                        (idx[1] in range(0, transformed_volume.GetHeight())) & \
                        (idx[2] in range(0, transformed_volume.GetDepth())):
                    # Pixel im neuen transformierten Volume speichern
                    pixel = volume.GetPixel(x, y, z)
                    transformed_volume.SetPixel(idx, pixel)

    return transformed_volume


def rotate_volume_y(volume, angle):
    """
     Rotiert das angegebene Volume um den angegebenen Winkel um die y-Achse.

     :param volume: Volume, welches um die y-Achse rotiert werden soll
     :param angle: Winkel, um wie viel Grad rotiert werden soll
     :returns: Das rotierte Volume
    """

    # Größe des Volumes speichern
    x_size = volume.GetWidth()
    y_size = volume.GetHeight()
    z_size = volume.GetDepth()

    # Spacing in x, y, z
    spacing = volume.GetSpacing()

    # mhd-Datei für das rotierte Volume erzeugen
    # Größe anpassen, d.h. Größe der x- und z-Achse tauschen (bei Numpy Array ist Reihenfolge z, y, x)
    rotated_volume = sItk.GetImageFromArray(np.full(shape=(x_size, y_size, z_size), fill_value=0, dtype=np.float32))

    # Spacing setzen (ebenfalls x- und z-Achse tauschen)
    rotated_volume.SetSpacing((spacing[2], spacing[1], spacing[0]))

    # Rotation um den angegebene Winkel um Y Achse festlegen
    rotation = sItk.VersorTransform((0, -1, 0), angle)

    # Drehpunkt festlegen, dieser ist so gewählt, dass der Ursprung links oben erhalten bleibt
    rotation.SetCenter((z_size * spacing[2] / 2, 0, z_size * spacing[2] / 2))

    # Rotation auf das angegebene Volume anwenden und rotiertes Volume zurückliefern
    return transform_voxel_of_volume(volume, rotated_volume, rotation)


def read_matrix_from_file(filename):
    """
     Liest aus der angegebenen Datei eine 4x4-Matrix ein, welche für eine affine Transformation verwendet werden kann.

     :param filename: Datei, aus der die Matrix eingelesen werden soll
     :returns: die eingelesene Matrix
    """

    matrix = []

    with open(filename) as input_file:

        # Matrix-Datei Zeilenweise einlesen und auf Gültigkeit prüfen
        for line in input_file:
            stripped_line = line.strip()

            # Bei jeder Zahl der Zeile prüfen, ob es ein numerischer Wert ist
            for number in stripped_line.split():
                try:
                    value = float(number)
                except ValueError:
                    print("Error: Matrix contains non numerical values.")
                    raise RuntimeError
                matrix += [value]

    # Für die Affine-Transformation muss es sich um eine 4x4 Matrix handeln
    if len(matrix) != 16:
        print("Error: Matrix size is not 4x4.")
        raise RuntimeError
    return matrix


def create_affine_transformation(matrix):
    """
    Erzeugt aus der gegebenen 4x4 Matrix ein Transformations-Objekt für die affine Transformation.

    :param matrix: Matrix für die affine Transformation
    :returns: Transformations-Objekt
    """

    # Affine Transformation erstellen
    transform_params = [matrix[0], matrix[1], matrix[2],
                        matrix[4], matrix[5], matrix[6],
                        matrix[8], matrix[9], matrix[10],
                        matrix[3], matrix[7], matrix[11]]
    affine = sItk.AffineTransform(3)
    affine.SetParameters(transform_params)

    return affine


def volume_registration(fixed_volume, moving_volume):
    """
    Führt eine Registrierung der beiden Volumes aus und liefert ein Transformations-Objekt zurück.

    :param fixed_volume: Referenz-Volume
    :param moving_volume: Volume auf das die affine Transformation ausgeführt wird
    :returns: Transformations-Objekt für die anzuwendende affine Transformation
    """

    registration = sItk.ImageRegistrationMethod()
    # Similarity = Prinzip zur Bestimmung der Ähnlichkeit
    registration.SetMetricAsCorrelation()
    # Optimierer
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50, convergenceMinimumValue=1e-6,
                                               convergenceWindowSize=10)

    # Art der Transformation festlegen
    registration.SetInitialTransform(sItk.AffineTransform(fixed_volume.GetDimension()))
    # Art der Interpolation festlegen, da ggf. einige Pixel interpoliert werden müssen
    registration.SetInterpolator(sItk.sitkLinear)
    # Ausführung der Registrierung
    affine_trans = registration.Execute(fixed_volume, moving_volume)

    # Ausgabe der Ergebnisse der Registrierung
    print(f"Registration optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {registration.GetOptimizerIteration()}")
    print(f" Metric value: {registration.GetMetricValue()}")
    return affine_trans


def fuse_volumes(ref_vol, cor_vol, sag_vol, transformation):
    """
     Fusioniert die koronale und die sagittale Ansicht und schreibt die Pixel beider Volumes in ein neues
     Volume, welches zurückgeliefert wird. Für die Fusion soll eine affine Transformation auf das sagittale
     Volume angewendet werden, damit die Volumes "besser" aufeinander passen und die anschließende Interpolation
     ein verbessertes Ergebnis erzeugen kann. Die affine Transformation wird durch die übergebene Matrix angegeben.

     :param ref_vol: Referenz-Volume, dient zum Festlegen der Maße des Ergebnis-Volumes
     :param cor_vol: Koronale Ansicht
     :param sag_vol: Sagittale Ansicht
     :param transformation: Die affine Transformation, die auf die Sagittale Ansicht angewendet wird
     :returns: Das fusionierte Volume
    """

    # Spacing für das Ergebnisbild ermitteln (koronale Ansicht als Referenz)
    spacing = ref_vol.GetSpacing()[0]

    # Größe in x- und y-Richtung für das Ergebnisbild festlegen
    size = min(ref_vol.GetWidth(), ref_vol.GetHeight())
    # Anzahl Slices ermitteln, damit die Voxelgröße in x, y und z gleich ist
    # Slice hinten soll ein vollständiges koronales Bild sein, deshalb - 1
    z_size = round((ref_vol.GetDepth() - 1) * ref_vol.GetSpacing()[2] / spacing) + 1

    # mhd-Datei für die fusionierten Ansichten erzeugen (koronale Ansicht als Referenz)
    fused_vol = sItk.GetImageFromArray(np.full(shape=(z_size, size, size), fill_value=np.nan, dtype=np.float32))
    # Spacing anpassen (überall gleich)
    fused_vol.SetSpacing((spacing, spacing, spacing))

    # Affine Transformation auf das sagittale Volume anwenden und Pixel im fusionierten Volume speichern
    fused_vol = transform_voxel_of_volume(sag_vol, fused_vol, transformation)

    # Pixel des koronalen Volumes im fusionierten Volume speichern
    fused_vol = transform_voxel_of_volume(cor_vol, fused_vol, None)

    return fused_vol


def interpolate_volume_3d(volume, num_of_cubes, fast_interpolation):
    """
     Interpoliert das fusionierte Volume, welches die koronale und die sagittale Ansicht enthält.
     Dabei wird eine dreidimensionale Interpolation angewendet.

     :param volume: Volume, welches interpoliert werden soll
     :param num_of_cubes: Anzahl der Quadrate pro Kante, in die das Volume aufgeteilt wird
     :param fast_interpolation: Wenn True, wird für die Interpolation Nearest Neighbor verwendet
     :returns: Das fusionierte und interpolierte Volume
    """

    # Größe des Volumes speichern
    x_size = volume.GetWidth()
    y_size = volume.GetHeight()
    z_size = volume.GetDepth()

    # y-Achse durchlaufen und Pixelwerte am Rand setzen (notwendig für die Interpolation)
    for y in range(0, y_size):
        # Pixelwerte an den z-Rändern auf 0 setzen
        for x in range(0, x_size):

            if np.isnan(volume.GetPixel((x, y, 0))):
                volume.SetPixel((x, y, 0), 0.0)

            if np.isnan(volume.GetPixel((x, y, z_size - 1))):
                volume.SetPixel((x, y, z_size - 1), 0.0)

        # Pixelwerte an den x-Rändern auf 0 setzen
        for z in range(0, z_size):

            if np.isnan(volume.GetPixel((0, y, z))):
                volume.SetPixel((0, y, z), 0.0)

            if np.isnan(volume.GetPixel((x_size - 1, y, z))):
                volume.SetPixel((x_size - 1, y, z), 0.0)

    # Anzahl der Pixel pro cube für jede Achse
    y_step = round(np.ceil(y_size / num_of_cubes))
    x_step = round(np.ceil(x_size / num_of_cubes))
    z_step = round(np.ceil(z_size / num_of_cubes))

    # Anzahl der Pixel pro Achse die zur Berechnung der Interpolation zusätzlich verwendet werden
    ovs = 5

    # y_min,y_max, x_min,x_max und z_min,zmax bestimmen den Quader der zu interpolierenden Punkte eines Cubes
    
    # y_calc_min, y_calc_max, x_calc_min, x_calc_max, z_calc_min, z_calc_max bestimmen den Quader der Punkte,
    # die für die Interpolation des Cubes herangezogen werden

    y_max = 0
    while y_max < y_size:

        y_min = y_max
        y_max += y_step
        if y_max > y_size:
            y_max = y_size

        y_calc_min = y_min - ovs
        y_calc_max = y_max + ovs
        if y_calc_min < 0:
            y_calc_min = 0
        if y_calc_max > y_size:
            y_calc_max = y_size

        x_max = 0
        while x_max < x_size:

            x_min = x_max
            x_max += x_step
            if x_max > x_size:
                x_max = x_size

            x_calc_min = x_min - ovs
            x_calc_max = x_max + ovs
            if x_calc_min < 0:
                x_calc_min = 0
            if x_calc_max > x_size:
                x_calc_max = x_size

            z_max = 0
            while z_max < z_size:

                z_min = z_max
                z_max += z_step
                if z_max > z_size:
                    z_max = z_size

                z_calc_min = z_min - ovs
                z_calc_max = z_max + ovs
                if z_calc_min < 0:
                    z_calc_min = 0
                if z_calc_max > z_size:
                    z_calc_max = z_size

                # Insgesamt werden 2 Durchläufe pro Cube zugelassen
                # Im ersten Durchlauf werden alle nicht vorhandenen Voxel durch lineare Interpolation bestimmt
                # Können am Rand die Werte nicht ermittelt werden, werden diese in einem 2. Durchlauf durch
                # eine nearest neighbor Interpolation bestimmt
                # Wenn fast_interpolation gesetzt ist, wird gleich mit dem 2. Durchlauf begonnen
                if not fast_interpolation:
                    tries = 1
                else:
                    tries = 2

                while tries <= 2:
                    num_of_def = 0
                    num_of_undef = 0

                    for y in range(y_calc_min, y_calc_max):
                        for x in range(x_calc_min, x_calc_max):
                            for z in range(z_calc_min, z_calc_max):
                                pixel = volume.GetPixel((x, y, z))
                                # Anzahl vorhandener und zu ermittelnder Pixel bestimmen
                                if (z in range(z_min, z_max)) & (y in range(y_min, y_max)) & (x in range(x_min, x_max)):
                                    if np.isnan(pixel):
                                        num_of_undef += 1
                                if not np.isnan(pixel):
                                    num_of_def += 1

                    # Wenn nach einem Durchlauf keine Punkte mehr zu interpolieren übrig sind,
                    # wird die while Schleife beendet
                    if num_of_undef == 0:
                        break

                    # Array für die Koordinaten der vorhandenen Pixelwerte pro cube anlegen,
                    # alle Punkte mit x, y, z speichern
                    defined_points = np.full(shape=(num_of_def, 3), fill_value=0, dtype=np.float64)
                    # Array für die vorhandenen Pixelwerte anlegen
                    values = np.full(shape=num_of_def, fill_value=0, dtype=np.float64)
                    # Array für die Koordinaten der zu ermittelnden Pixelwerte pro cube anlegen,
                    # alle Punkte mit x, y, z speichern
                    undefined_points = np.full(shape=(num_of_undef, 3), fill_value=0, dtype=np.float64)

                    i = 0
                    j = 0
                    for y in range(y_calc_min, y_calc_max):
                        for x in range(x_calc_min, x_calc_max):
                            for z in range(z_calc_min, z_calc_max):

                                # Aktueller Pixelwert
                                pixel = volume.GetPixel((x, y, z))
                                if (z in range(z_min, z_max)) & (y in range(y_min, y_max)) & (x in range(x_min, x_max)):
                                    # Wenn Pixel undefinierten Wert hat
                                    if np.isnan(pixel):
                                        # zu ermittelnden Punkt mit x, y, z speichern
                                        undefined_points[j] = (x, y, z)
                                        j += 1
                                if not np.isnan(pixel):
                                    # vorhandene Punkte mit x, z und Pixelwert speichern
                                    defined_points[i] = (x, y, z)
                                    values[i] = pixel
                                    i += 1

                    # Interpolation mithilfe der Arrays (für jeden cube, alle interpolierten Pixelwerte speichern)
                    if tries == 1:  # Im ersten Durchlauf werden die Voxel linear interpoliert
                        interpolated_vals = scipy.interpolate.griddata(defined_points, values, undefined_points,
                                                                       method='linear')
                    else:  # Im 2. Durchlauf bzw. wenn fast_interpolation aktiv ist,
                        # werden die Voxel durch nearest neighbor Interpolation ermittelt
                        interpolated_vals = scipy.interpolate.griddata(defined_points, values, undefined_points,
                                                                       method='nearest')

                    for i in range(0, num_of_undef):
                        # Koordinaten ermitteln (Waren zuvor unbestimmt)
                        x = round(undefined_points[i][0])
                        y = round(undefined_points[i][1])
                        z = round(undefined_points[i][2])

                        # Pixelwert an der jeweiligen Position speichern
                        pixel = interpolated_vals[i]
                        volume.SetPixel((x, y, z), pixel)
                    tries += 1

    # Datei mit den fusionierten Ansichten zurückliefern
    return volume


def interpolate_volume_2d(volume, fast_interpolation):
    """
     Interpoliert das fusionierte Volume, welches die koronale und die sagittale Ansicht enthält.
     Für die Interpolation wird zunächst sichergestellt, dass überall am Rand Werte vorhanden sind.
     Die vorhandenen und zu ermittelnden Pixel werden pro xz-Ebene gespeichert und durch die Interpolation werden
     die undefinierten Pixel ermittelt.

     :param volume: Volume, welches interpoliert werden soll
     :param fast_interpolation: Wenn True, wird für die Interpolation Nearest Neighbor verwendet
     :returns: Das fusionierte und interpolierte Volume
    """

    # Größe des Volumes speichern
    x_size = volume.GetWidth()
    y_size = volume.GetHeight()
    z_size = volume.GetDepth()

    # Verarbeitung/Interpolation jeder xz-Ebene
    for y in range(0, y_size):

        num_of_def = 0

        # Für jede xz-Ebene:
        # Anzahl definierter/undefinierter Pixel bestimmen
        # Pixelwerte am Rand der xz-Ebene setzen (notwendig für die Interpolation)
        for x in range(0, x_size):
            for z in range(0, z_size):
                # Aktueller Pixelwert
                pixel = volume.GetPixel((x, y, z))

                # Pixelwerte an den x-Rändern auf 0 setzen
                if ((x == 0) | (x == x_size - 1)) & (np.isnan(pixel)):
                    volume.SetPixel((x, y, z), 0.0)

                # Pixelwerte an den z-Rändern auf 0 setzen
                if ((z == 0) | (z == z_size - 1)) & (np.isnan(pixel)):
                    volume.SetPixel((x, y, z), 0.0)

                # Anzahl definierter Pixel bestimmen
                if not (np.isnan(volume.GetPixel((x, y, z)))):
                    num_of_def += 1

        # Anzahl undefinierter Pixel berechnen
        num_of_undef = (x_size * z_size) - num_of_def

        # Array für die vorhandenen Pixelwerte anlegen (für jede xz-Ebene, alle Punkte mit x, z und Pixelwert speichern)
        defined_vals = np.full(shape=(num_of_def, 2), fill_value=np.nan, dtype=np.float64)
        # Array für die vorhandenen Pixelwerte anlegen
        values = np.full(shape=num_of_def, fill_value=0, dtype=np.float64)
        # Array für die zu ermittelnden Pixelwerte anlegen (für jede xz-Ebene, alle Punkte mit x und z speichern)
        undefined_vals = np.full(shape=(num_of_undef, 2), fill_value=np.nan, dtype=np.float64)

        #  Arrays für die Interpolation befüllen
        i = 0
        j = 0
        # Für jede xz-Ebene die definierten und undefinierten Pixel in den Arrays speichern
        for x in range(0, x_size):
            for z in range(0, z_size):
                # Aktueller Pixelwert
                pixel = volume.GetPixel((x, y, z))

                # Wenn Pixel undefinierten Wert hat
                if np.isnan(pixel):
                    # Pixel auf 0 setzen
                    volume.SetPixel((x, y, z), 0.0)
                    # zu ermittelnden Punkt mit x, z speichern
                    undefined_vals[j] = (x, z)
                    j += 1
                else:
                    # vorhandene Punkte mit x, z und Pixelwert speichern
                    defined_vals[i] = (x, z)
                    values[i] = pixel
                    i += 1

        # Interpolation mithilfe der Arrays (für jede xz-Ebene, alle interpolierten Pixelwerte speichern)
        if not fast_interpolation:
            interpolated_vals = scipy.interpolate.griddata(defined_vals, values, undefined_vals,
                                                           method='cubic')
        else:
            interpolated_vals = scipy.interpolate.griddata(defined_vals, values, undefined_vals,
                                                           method='nearest')

        # Die unbestimmten Werte der xz-Ebene durchlaufen und die ermittelten Werte im Volume speichern
        for i in range(0, num_of_undef):
            # x- und z-Koordinaten ermitteln (Waren zuvor unbestimmt)
            x = round(undefined_vals[i][0])
            z = round(undefined_vals[i][1])

            # Pixelwert an der jeweiligen Position speichern
            pixel = interpolated_vals[i]
            volume.SetPixel((x, y, z), pixel)

    # Datei mit den fusionierten Ansichten zurück liefern  ----------
    return volume


if __name__ == '__main__':

    try:
        # ---------- Parameter von der Kommandozeile einlesen ----------
        cor_file, sag_file, result_file, no_registration, matrix_file, interpolation3d, output_dir, \
            do_fast_interpolation = read_params()

        # ---------- Einlesen der beiden Datensätze (Ansichten) ----------

        # Prüfen, ob die Ansichten zusammenpassen
        cor_volume, sag_volume = read_and_check_volumes(cor_file, sag_file)

        # ---------- Vorverarbeitung der Volumes ----------
        affine_transformation = sItk.AffineTransform(3)

        # Wenn sich das X oder Y Spacing des Sagittalen Volumen vom Spacing des Koronalen Volumen unterscheidet,
        # wird das Sagittale Volumen im X und Y Spacing an das Spacing des Koronalen Volumens angepasst.
        # Da Spacings als float Zahlen angegeben sind, erfolgt der Vergleich über Differenz und Toleranzwert Epsilon
        if ((abs(cor_volume.GetSpacing()[0] - sag_volume.GetSpacing()[0]) > 0.001) or
                (abs(cor_volume.GetSpacing()[1] - sag_volume.GetSpacing()[1]) > 0.001)):
            print("Resampling sagittal volume to adjust spacing...")
            newSpacing = (cor_volume.GetSpacing()[0], cor_volume.GetSpacing()[1], sag_volume.GetSpacing()[2])
            sag_volume = resample_volume_to_new_voxel(sag_volume, newSpacing, do_fast_interpolation)

        # Sagittales Volume um 90° um die y-Achse rotieren (notwendig, damit die Ansichten zusammenpassen)
        print("Rotating sagittal volume...")
        sag_volume_rotated = rotate_volume_y(sag_volume, np.pi / 2)

        # Berechnung neuer Auflösung ist nur notwendig, wenn noch eine Registrierung stattfinden soll
        if matrix_file is None:
            # Koronales Volume auf IsoVoxelSize bringen, damit die Voxelgröße in x, y und z gleich ist
            print("Resampling coronal volume to isometric voxel size...")
            # Spacing ist durch die x-Achse gegeben (koronale Ansicht als Referenz)
            cor_volume_resampled = resample_volume_to_iso_voxel(cor_volume, cor_volume.GetSpacing()[0],
                                                                do_fast_interpolation)

            # Sagittales gedrehtes Volume auf IsoVoxelSize bringen, damit die Voxelgröße in x, y und z gleich ist
            # Spacing wie für das Koronale Volume verwenden, damit die Pixel zusammenpassen
            print("Resampling rotated sagittal volume to isometric voxel size...")
            sag_volume_resampled = resample_volume_to_iso_voxel(sag_volume_rotated, cor_volume.GetSpacing()[0],
                                                                do_fast_interpolation)

            # ---------- Registrierung der Volumes (Matrix ermitteln) ----------

            # Wenn der User keine Registrierung (externe Registrierung) angegeben hat - Programm beenden
            if no_registration:
                out_cor_file = os.path.join(os.path.abspath(output_dir), "CorVolume4Registration.mhd")
                print("Writing coronal volume for using with preferred image registration: " + out_cor_file + "...")
                sItk.WriteImage(cor_volume_resampled, out_cor_file)

                out_sag_file = os.path.join(os.path.abspath(output_dir), "SagVolume4Registration.mhd")
                print("Writing sagittal volume for using with preferred image registration: " + out_sag_file + "...")
                sItk.WriteImage(sag_volume_resampled, out_sag_file)

                print("\nRestart this program with the -m option after the registration is done to finish the merge.")
                exit(0)

            # Wenn noch eine Registrierung stattfinden soll, aber keine Matrix angegeben ist (noch keine Registrierung)
            else:
                # Wird eine Registrierung durchgeführt und eine 4x4 Matrix erzeugt für die Transformation
                print("Generating 4x4 matrix for affine transformation by performing a 3D image registration...")
                affine_transformation = volume_registration(cor_volume_resampled, sag_volume_resampled)

        # Wenn der User bereits eine Registrierung durchgeführt hat, wird die Matrix für die Transformation eingelesen
        else:
            print("Reading 4x4 matrix for affine transformation: " + matrix_file + "...")
            affine_matrix = read_matrix_from_file(matrix_file)
            affine_transformation = create_affine_transformation(affine_matrix)

        # ---------- Fusion der Volumes ----------

        # Fusioniert die beiden Volumes mithilfe der Transformationsmatrix, die durch die Registrierung erzeugt wurde
        print("Merging sagittal volume into coronal volume using affine transformation...")
        fused_volume = fuse_volumes(cor_volume, cor_volume, sag_volume_rotated, affine_transformation)

        # Für Debug Zwecke ist es manchmal hilfreich das FusedVolume zu speichern
        """
        print("Writing fused volume...")
        out_file = os.path.join(os.path.abspath(output_dir), "FusedVolume.mhd")
        sItk.WriteImage(fused_volume, out_file)
        """

        # ---------- Interpolation des Volumes ----------
        if interpolation3d > 0:
            # Die fehlenden Pixel des neuen Volumes werden durch eine 3D-Interpolation ergänzt
            print("Interpolating missing voxel in merged result volume by 3D Interpolation...")
            result_volume = interpolate_volume_3d(fused_volume, interpolation3d, do_fast_interpolation)
        else:
            # Die fehlenden Pixel des neuen Volumes werden durch eine kubische 2D-Interpolation ergänzt
            print("Interpolating missing voxel in merged result volume by 2D Interpolation...")
            result_volume = interpolate_volume_2d(fused_volume, do_fast_interpolation)

        # ---------- Ergebnis schreiben  ----------
        print("Writing merged result volume to output file: " + result_file + "...")
        sItk.WriteImage(result_volume, result_file)

    except RuntimeError:
        exit(1)
