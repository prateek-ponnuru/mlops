from time import sleep
from random import randint, sample, uniform
from ref_data import cities, degrees, experience, skill_list
import requests


def gen_row():

    exper = randint(experience[0], experience[1])
    exper += uniform(0, 1)
    exper = round(exper * 2) / 2

    city = randint(0, len(cities)-1)
    city = cities[city]

    degree = randint(0, len(degrees)-1)
    degree = degrees[degree]

    skills = sample(skill_list, randint(3, 15))

    if uniform(0, 1) > 0.95:
        non_existing_skill = 'pascal'
        skills.append(non_existing_skill)

    row = {'diplome': degree,
           'ville': city,
           'entreprise': 'UNKNOWN',
           'experience': exper,
           'skills': skills}

    return row


with open("data/data_sim.csv", 'w') as f_target:
    f_target.write(f"Entreprise,Diplome,Ville,Experience,Technologies,Techlist\n")
    for i in range(100):
        row = gen_row()
        f_target.write(f",{row['entreprise']},{row['diplome']},"
                       f"{row['ville']},{row['experience']},"
                       f"{'/'.join(row['skills'])},"
                       f"\"{row['skills']}\"\n")

        resp = requests.request("POST",
                                "http://18.157.175.19:9696/predict/",
                                json=row)
        print(f"prediction: {resp.json()['prediction']}")
    # sleep(1)