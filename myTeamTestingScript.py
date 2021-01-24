import subprocess
import random
from subprocess import Popen
import time
import pandas as pd

times = 5
seeds = set([random.randint(1,10000) for i in range(times)])
output = []
TEAM_RED = "myTeam.py"
teams = ["baseLineTeam.py"]
Scores = []

def updateScore(text,TEAM_BLUE,SEED):
    scoreRow = [TEAM_RED,TEAM_BLUE,f'RANDOM{SEED}']
    result = ""
    if 'Tie' in text:
        scoreRow.append('Tie')
        Scores.append(scoreRow)
    elif 'Red' in text:
        scoreRow.append(TEAM_RED)
        Scores.append(scoreRow)
    elif 'Blue' in text:
        scoreRow.append(TEAM_BLUE)
        Scores.append(scoreRow)
    

for i, SEED in enumerate(seeds):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    for TEAM_BLUE in teams:
        print(f"====== Running Game: RANDOM{SEED} - {TEAM_RED} vs {TEAM_BLUE} ======")
        path_to_output_file = f'logs/log_{timestr}_RANDOM{SEED}_{TEAM_BLUE.split(".")[0]}.txt'
        myoutput = open(path_to_output_file, 'w+')
        proc = Popen(f'python capture.py -r {TEAM_RED} -b {TEAM_BLUE} -l RANDOM{SEED} -q'.split(),
                            stdout=myoutput, stderr=subprocess.PIPE, universal_newlines=True)
        proc.wait()
        with open(path_to_output_file, "r") as file:
            for text in (file.readlines()[-3:]):
                if len(text.strip()) > 0:
                    out = f'{text.strip()}'
                    updateScore(out,TEAM_BLUE,SEED)
                    print(out)

df = pd.DataFrame(Scores, columns=['Red_Team','Blue_Team','Map','Result'])
print(df)
timestr = time.strftime("%Y%m%d_%H%M%S")
df.to_csv(f'logs/logs_{timestr}.csv')