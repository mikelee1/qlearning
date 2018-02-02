from nnmodel import *
from environment import *
statenum = 20
actions = ['left', 'right']
episode = 2000
epsilon =1
delt = 0.9
robot = ['_'] * statenum
dic = {1: 'left', 0: 'right'}


def train(qtable):
    currentstate = 0
    trainlist = []
    env = envstate(statenum,currentstate)
    for j in range(episode):
        treanum = env.reset()   # initiate
        print(treanum)
        while True:
            env.refresh()
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = qtable.evaluate([[currentstate, treanum]])
                action = dic[action[0]]

            r, s_n, terminal = env.step(currentstate, action, treanum, statenum)
            if r > 0:
                trainlist.append([currentstate, treanum, action])
            if terminal:

                break
            currentstate = s_n
        qtable.train(trainlist)



def evaluate(qtable):
    currentstate = 0
    env = envstate(statenum,currentstate)
    treanum = env.reset()
    while True:
        env.refresh()
        action = qtable.evaluate([[currentstate, treanum]])
        action = dic[action[0]]
        r, s_n, terminal = env.step(currentstate, action, treanum, statenum)
        if terminal:
            print('\n')
            break
        currentstate = s_n


def main():
    qtable = model(actions, statenum)
    train1 = False
    if train1:
        train(qtable)
    else:
        evaluate(qtable)


if __name__ == '__main__':
    main()
