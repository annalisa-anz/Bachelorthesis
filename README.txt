@autor Annalisa Anzioso 31.08.2022


Beschreibung:
--------------------
Das Programm "volumeMerger.py" dient dazu, zwei Bilddatensätze aus unterschiedlichen Perspektiven zu fusionieren und so die Bildauflösung zu erhöhen.
Es wird als Parameter des Python Interpreters über die Kommandozeile ausgeführt und stellt verschiedene Parameter zur Verfügung, mit denen es aufgerufen werden kann.
Die ersten beiden Parameter geben die koronale und sagittale Ansicht an, welche fusioniert werden sollen. Alle weiteren Parameter sind optional anzugeben.
Das erzeugte Ergebnis ist ein Volumen mit isotropen Voxeln, dass die Informationen beider Bilddatensätze enthält. 


Notwendige Parameter:
--------------------
- koronal_file:  Datei mit dem koronalen Volumen, dass fusioniert werden soll
- sagittal_file: Datei mit dem sagittalen Volumen, dass fusioniert werden soll


Optionale Parameter:
--------------------
- "-r", "--resultfile":      Angabe einer Datei in die das Ergebnis geschrieben wird
- "-n", "--noregistration":  keine Registrierung im Programm, nur Vorverarbeitung für eine anderweitige Registrierung
- "-m", "--matrixfile":      Angabe einer Datei, die die generierte Matrix aus einer vorherigen Registrierung enthält
- "-i", "--interpolation3d": Angabe, dass eine 3D-Interpolation durchgeführt werden soll (standardmäßig 2D-Interpolation)
- "-o", "--outputdir":       Angabe eines Pfades für die generierten Dateien
- "-f", "--fast":            Angabe, dass bei der Interpolation die Nearest Neighbor Methode genutzt werden soll


Externe Abhängigkeiten:
--------------------
- Python    > 3.9
- SimpleITK > 2.1
- NumPy     > 1.23
- SciPy     > 1.8





