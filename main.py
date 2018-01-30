import pandas as pd
import numpy as np
import random
import time,os
statenum = 10
actions = ['left','right']
episode = 40
epsilon = 0.99
delt = 0.9
treanum = 9
def build_qtabel(actions,statenum):
    print(actions)
    a = np.zeros([statenum,len(actions)])
    df = pd.DataFrame(a,columns=actions)
    return df

def gene_action(s,qtable):
    # if s == 6:
    #     time.sleep(1)
    k = random.random()
    if k<epsilon and qtable.iloc[s,:].all()!=0:

        action = qtable.iloc[s,:].argmax()

    else:

        action = random.choice(actions)

    return action


def step(s,action):
    r = 0
    terminal = False
    print(action)
    if action == 'left':

        s_n = s-1
        if s_n == -1:
            r = -1
            terminal = True

    if action == 'right':
        s_n = s + 1
        if s_n == statenum-1:
            r = 5
            terminal = True
    print(s_n)
    #s_ = min(max(0,s_),5)
    return r,s_n,terminal



def update_qtable(qtable,s,s_,a,r):
    if s_ !=treanum:
        qtable.iloc[s,:][a] += 0.1*(r+ delt*qtable.iloc[s_,:].max() - qtable.iloc[s,:][a])
    else:
        qtable.iloc[s,:][a] = r


robot = ['_']*statenum
def main():
    a = build_qtabel(actions, statenum)
    if os.path.exists('qtable.csv'):
        a = pd.read_csv('qtable.csv')

    for i in range(episode):
        s = 0
        while True:
            robot = ['_'] * statenum
            action = gene_action(s,a)
            robot[s]='+'
            print(''.join(robot))
            r,s_,terminal = step(s,action)

            if terminal:

                update_qtable(a,s,s_,action,r)
                print(a)
                break
            else:
                update_qtable(a,s,s_,action,r)
            time.sleep(0.1)
            s = s_

    a.to_csv('qtable.csv',index=False)


if __name__ == '__main__':
    main()