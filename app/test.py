import os
import shutil

faliUser = [ "Abdullah_Gul",
             "Abel_Pacheco",
             "Abid_Hamid_Mahmud_Al-Tikriti",
             "Adam_Scott",
             "Adolfo_Rodriguez_Saa",
             "Adrian_McPherson",
             "Ahmad_Masood",
             "Ahmed_Chalabi",
             "Ahmet_Necdet_Sezer",
             "Akbar_Hashemi_Rafsanjani",
             "Alan_Mulally",
             "Alastair_Campbell",
             "Alberto_Fujimori",
             "Alberto_Ruiz_Gallardon",
             "Albrecht_Mentz",
             "Aldo_Paredes",
             "Alec_Baldwin",
             "Alejandro_Avila",
             "Alejandro_Toledo",
             "Alexander_Downer",
             "Alexandra_Stevenson",
             "Alexandra_Vodjanikova",
             "Alex_Barros",
             "Alicia_Silverstone",
             "Alison_Lohman",
             "Ali_Abbas",
             "Allison_Janney",
             "Allyson_Felix",
             "Alvaro_Noboa",
             "Alvaro_Silva_Calderon",
             "Alvaro_Uribe",
             "Al_Sharpton",
             "Ana_Guevara",
             "Andrew_Bunner",
             "Andrew_Cuomo",
             "Andrew_Weissmann",
             "Andre_Agassi",
             "Angelina_Jolie",
             "Anna_Kournikova",
             "Anna_Nicole_Smith",
             "Anneli_Jaatteenmaki",
             "Anthony_Hopkins",
             "Anwar_Ibrahim",
             "Arianna_Huffington",
             "Ariel_Sharon",
             "Ari_Fleischer",
             "Arlen_Specter",
             "Arminio_Fraga",
             "Arnaud_Clement",
             "Aron_Ralston" ]


directory = 'E:\\99.lmk810501\\13.AI\\deepface-master\\tests\\dataset3'
extension = 'jpg'

i = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        isFail = False
        if file.endswith(extension):
            file_path = os.path.join(root, file)
            dir_path = os.path.dirname(file_path)
            if '0002' in file_path:
                for filename in faliUser:
                    if filename + '_0002' in file_path:
                        isFail = True

                if isFail:
                    shutil.copy(file_path, 'E:\\99.lmk810501\\13.AI\\deepface-master\\tests\\dataset_fail')
                else:
                    shutil.copy(file_path, 'E:\\99.lmk810501\\13.AI\\deepface-master\\tests\\dataset_success')






