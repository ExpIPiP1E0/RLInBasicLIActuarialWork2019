# model inspection
from keras import models
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def inspect_hidden_layers(env, model, agent, show_hidden_hist=False, figsize=(15, 2), *args, **kargs):
    layers_input = model.inputs
    layers_hidden = [layer.output for layer in model.layers if layer.input not in model.inputs]
    model_inspection = models.Model(layers_input, layers_hidden)

    env.reset()

    for t in range(20):
        print('-----------------------------------------------------------')
        print('t = ', t)
        state = env.state()
        preds = model_inspection.predict([*state])
        action = agent.get_action(state, greedy=True)
        statt_next, reward, _, done = env.step(action)

        for i, state_comp in enumerate(state):
            print('state {} min = {} state {} max = {}'
                  .format(i, np.min(state_comp), i, np.max(state_comp)))

        print('action min = ', np.min(action),
              'action max = ', np.max(action))

        print('reward min = ', np.min(reward),
              'reward max = ', np.max(reward))

        stat_dict = {'layer': [], 'min': [], 'max': []}
        if show_hidden_hist:
            fig = plt.figure(figsize=figsize)
        for i, pred in enumerate(preds):

            # print('layer = {}, min = {}, max = {}'.format(layers_hidden[i].name, np.min(pred), np.max(pred)))
            stat_dict['layer'].append(layers_hidden[i].name)
            stat_dict['min'].append(np.min(pred))
            stat_dict['max'].append(np.max(pred))
            if show_hidden_hist:
                ax = fig.add_subplot(1, len(preds), 1 + i)
                ax.hist(pred[0], bins=20)
        print(pd.DataFrame(data=stat_dict))
        if show_hidden_hist:
            plt.tight_layout()
            plt.show()

        print('')

        if done[0, 0]:
            print('END by done')
            break


if __name__=='__main__':
    inspect_hidden_layers(tester.env, tester.model, tester.agent)