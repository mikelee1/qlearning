import pandas as pd
import numpy as np
import random
import time,os
statenum = 20
actions = ['left','right']
episode = 1000
epsilon = 1#0.99
delt = 0.9
treanum = 9
def build_qtabel(actions,statenum):
    #statenum 10
    #actions ['left' 'right']


    data = np.zeros([statenum*statenum, 2])

    if os.path.exists('qtable.csv'):
        data = np.loadtxt('qtable.csv',dtype=np.float32, delimiter=",")

    d = sum([[i]*statenum for i in range(statenum)],[])
    f = sum([[i] for i in range(statenum)] * statenum,[])
    arrays = [np.array(d), np.array(f)]
    df = pd.DataFrame(data, index=arrays,columns=['left','right'])

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


def step(s,action,treanum,statenum):
    r = 0
    terminal = False

    if action == 'left':

        s_n = s-1
        if s_n == treanum:
            r = 5
            terminal = True
        elif s_n == -1:
            r = -1
            terminal = True

    if action == 'right':
        s_n = s + 1
        if s_n == treanum:
            r = 5
            terminal = True
        elif s_n >= statenum:
            terminal = True
            r = -1

    #s_ = min(max(0,s_),5)
    return r,s_n,terminal



def update_qtable(qtable,s,s_,a,r,treanum):
    if s_ !=treanum:
        if s_ == statenum or s_ == -1:
            qtable.iloc[s,:][a] = -1
        else:
            qtable.iloc[s,:][a] += 0.1*(r+ delt*qtable.iloc[s_,:].max() - qtable.iloc[s,:][a])
    else:
        qtable.iloc[s,:][a] = r


robot = ['_']*statenum
def main():
    qtable = build_qtabel(actions, statenum)

    s = 0
    for i in range(episode):
        #s = 0
        # if s== 19:
        #     raise ValueError
        treanum = random.randint(0,statenum-1)
        #treanum = 5
        print(treanum)
        a = qtable.loc[treanum]

        while True:
            if s == treanum:
                break
            robot = ['_'] * statenum
            action = gene_action(s,a)
            robot[treanum]='O'
            robot[s]='+'
            print(''.join(robot))
            r,s_,terminal = step(s,action,treanum,statenum)

            if terminal :
                #print('upupup',s_)
                update_qtable(a,s,s_,action,r,treanum)

                break
            else:
                update_qtable(a,s,s_,action,r,treanum)
            #time.sleep(0.3)
            s = s_

    qtable.to_csv('qtable.csv',index=False,header=False)


if __name__ == '__main__':
    main()