import pandas as pd
import numpy as np
import random
import time,os
from nnmodel import *


statenum = 20
actions = ['left','right']
episode = 10
epsilon = 0.01#0.99
delt = 0.9
treanum = 9
def build_qtabel(actions,statenum):
    data = np.zeros([statenum*statenum, 2])

    if os.path.exists('qtable.csv'):
        data = np.loadtxt('qtable.csv',dtype=np.float32, delimiter=",")

    d = sum([[i]*statenum for i in range(statenum)],[])
    f = sum([[i] for i in range(statenum)] * statenum,[])
    arrays = [np.array(d), np.array(f)]
    df = pd.DataFrame(data, index=arrays,columns=['left','right'])

    return df

def gene_action(s,qtable):

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


dic = {1:'left',0:'right'}
def train(qtable):
    currentstate = 0
    trainlist = []
    for i in range(episode):
        treanum = random.randint(0, statenum - 1)
        print(treanum)
        while True:
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                state =[[currentstate,treanum]]
                action = qtable.evaluate(state)
                action = dic[action[0]]
                #print(state,action)


            robot = ['_'] * statenum

            robot[treanum]='O'
            robot[currentstate]='+'
            print(''.join(robot))

            r, s_n, terminal = step(currentstate, action, treanum, statenum)
            if terminal:
                if r>0:
                    trainlist.append([currentstate,treanum,action])
                break
            currentstate = s_n


        #qtable.train(trainlist)
        if i %100 ==0:
            pass#qtable.save()
    #qtable.evaluate()
    #print(trainlist)



def evaluate(qtable):
    qtable.evaluate()



def main():
    qtable = model(actions,statenum)
    train1=True
    if train1:
        train(qtable)
    else:
        evaluate(qtable)


if __name__ == '__main__':
    main()
