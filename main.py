import os
from sklearn import model_selection
from tensorflow.keras import layers, models, metrics, losses, optimizers, callbacks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

class_list = [f'A{i:02}' for i in range(1, 21)] + ['R']


class HGP(object):
    def __init__(self, subj_name, w1, w2, start_thresh, save_detection_plots, ):
        self.subj_name = subj_name
        self.create_directories(CV=5)  # generate all directories for storage
        self.dataset_loader = DatasetLoader(subj_name, w1, w2, start_thresh=start_thresh,
                                            save_detection_plots=save_detection_plots,
                                            save_path=self.dataset_plots_dir)
        self.CV_datasets = self.dataset_loader.K_fold_dataset_split(CV=5)
        self.my_model = MyModel(self.subj_name, )

    def train(self):

        for i, cv in enumerate([f'cv_{i}' for i in range(1, 6)]):
            print(f'start to train model_{cv}!')
            self.my_model.initialize_model()
            self.my_model.train_model(train_X=self.CV_datasets[cv]['train']['X'],
                                      train_Y=self.CV_datasets[cv]['train']['Y'],
                                      test_X=self.CV_datasets[cv]['test']['X'],
                                      test_Y=self.CV_datasets[cv]['test']['Y'],
                                      model_name=cv,
                                      file_save_path=self.cv_results_save_dir + '/' + cv, epochs=300,
                                      EarlyStop_patience=30)

    def create_directories(self, CV=5):
        """create all directories for saving data/results"""
        self.dataset_plots_dir = f'results/{self.subj_name}/dataset_plots'
        os.makedirs(self.dataset_plots_dir, exist_ok=True)

        self.cv_results_save_dir = f'results/{self.subj_name}/CV_results/'
        for cv_index in range(1, CV + 1):
            os.makedirs(self.cv_results_save_dir + f'cv_{cv_index}', exist_ok=True)

    def test(self, decision_timestep_num_list=(10,20,30,40,50,60)):
        cv_list = [f'cv_{i}' for i in range(1,6)]
        from sklearn import metrics
        results_DF = pd.DataFrame(index=cv_list,columns=[f'acc_{num}' for num in decision_timestep_num_list])
        for decision_timestep_num in decision_timestep_num_list:
            for cv_index in [f'cv_{i}' for i in range(1,6)]:
                model_path = self.cv_results_save_dir + f'/{cv_index}/model_{cv_index}.h5'
                model: models.Model
                model = models.load_model(model_path)
                test_X = np.zeros_like(self.CV_datasets[cv_index]['test']['X'])
                test_sample_num = test_X.shape[0]

                for i in range(test_sample_num):
                    start = self.CV_datasets[cv_index]['test']['start'][i]
                    slice_1 = slice(start, start + decision_timestep_num)  # simulate online test with limited data points
                    slice_2 = slice(-1)  # offline test with all available data points, and its accuracy is higher than using slice_1
                    slice_ = slice_2
                    test_X[i][slice_, :] = self.CV_datasets[cv_index]['test']['X'][i][slice_, :]

                self.test_Y = self.CV_datasets[cv_index]['test']['Y']
                true_labels = []
                pred_labels = []
                self.outputs = model.predict(test_X)

                for sample_index in range(test_sample_num):
                    start = self.CV_datasets[cv_index]['test']['start'][sample_index]
                    pred_label = class_list[int(np.argmax(self.outputs[sample_index][start + decision_timestep_num, :]))]
                    pred_labels.append(pred_label)
                    true_label = class_list[int(np.argmax(self.test_Y[sample_index][start + decision_timestep_num, :]))]
                    true_labels.append(true_label)
                self.true = true_labels
                self.pred = pred_labels

                acc = metrics.accuracy_score(self.pred, self.true)
                results_DF.loc[cv_index,f'acc_{decision_timestep_num}'] = acc
        results_DF.to_csv(self.cv_results_save_dir+'/cv_acc_results.csv')


class DatasetLoader(object):
    """Load the dataset of single subject"""

    def __init__(self, subj_name: str, w1: int, w2: int, start_thresh: float, save_detection_plots, save_path,
                 class_num: int = 21, repeat_num: int = 30,
                 timestep_num: int = 400, channel_num: int = 8, ):
        """
        :param subj_name: Name of subject
        :param class_num: Number of classes of hand gestures
        :param repeat_num: Number of repetition of each kind of hand gesture
        :param timestep_num: Number of time steps of each repetition
        :param w1: Width of the first sliding window
        :param w2: Width of the second sliding window
        :param start_thresh: Threshold for detect motion start
        """
        self.subj_name = subj_name
        self.class_num = class_num
        self.repeat_num = repeat_num
        self.sample_num = class_num * repeat_num
        self.timestep_num = timestep_num
        self.channel_num = channel_num
        self.w1 = w1
        self.w2 = w2
        self.start_thresh = start_thresh
        self.save_detection_plots = save_detection_plots
        self.save_plots_path = save_path

        self.data = {'X': np.zeros(shape=(self.sample_num, timestep_num, channel_num), ),
                     'raw_Y': np.zeros(shape=(self.sample_num, timestep_num, class_num), dtype=int),
                     'Y': np.zeros(shape=(self.sample_num, timestep_num, class_num), dtype=int),
                     'start': np.zeros(shape=(self.sample_num,), dtype=int),
                     'end': np.zeros(shape=(self.sample_num,), dtype=int),
                     'S1': np.zeros(shape=(self.sample_num, timestep_num, channel_num), ),
                     'S2': np.zeros(shape=(self.sample_num, timestep_num))
                     }

        self.__load_data_from_txt()
        self.__detect_motion()
        self.__correct_labels()

    def __load_data_from_txt(self):
        """load txt files to a dict of dataset"""
        txt_path = f"datasets/{self.subj_name}"
        for sample_index, txt_file in enumerate(os.listdir("datasets/xx")):
            self.data['X'][sample_index] = np.loadtxt(f"{txt_path}/{txt_file}")
            self.data['raw_Y'][sample_index][:, sample_index // self.repeat_num] = 1
            self.data['Y'] = self.data['raw_Y'].copy()

    def __detect_motion(self, ):
        """"""

        for sample_index in range(self.sample_num):
            for timestep_index in range(1, self.timestep_num):
                if timestep_index - self.w1 < 0:
                    used_slice = slice(0, timestep_index + 1)
                else:
                    used_slice = slice(timestep_index - self.w1, timestep_index)
                self.data['S1'][sample_index][timestep_index, :] = np.std(self.data['X'][sample_index][used_slice, :],
                                                                          axis=0)
        self.data['S2'] = np.mean(self.data['S1'], axis=2)

        for sample_index in range(self.sample_num):
            if sample_index in range(self.sample_num - self.repeat_num,
                                     self.sample_num):  # sample is the gesture of rest, skip the sample.
                continue

            start_found = False
            greater_than_start_bool = self.data['S2'][sample_index, :] >= self.start_thresh
            for timestep_index in range(self.w1, self.timestep_num):  # skip few points at beginning.
                if np.all(greater_than_start_bool[timestep_index - self.w2:timestep_index]):  # start is found
                    if not start_found:
                        self.data['start'][sample_index] = timestep_index - self.w2 - round(
                            0.5 * self.w1)  # adjsut position according to these two sliding windows
                        start_found = True
                    self.data['end'][sample_index] = timestep_index  # update the end position all the time

        mpl.use('agg')
        if self.save_detection_plots:
            for sample_index in range(self.sample_num):
                fig, (ax1, ax2) = plt.subplots(2, 1)  # type: plt.Figure,(plt.Axes,plt.Axes)
                ax1.plot(self.data['X'][sample_index])
                ax2.plot(self.data['S2'][sample_index])
                ax2.axvline(x=self.data['start'][sample_index])
                ax2.axvline(x=self.data['end'][sample_index])
                fig.savefig(fname=self.save_plots_path + f'/{sample_index}.png')
                plt.close(fig)

    def __correct_labels(self):

        for sample_index in range(self.sample_num):
            start = self.data['start'][sample_index]
            end = self.data['end'][sample_index]
            self.data['Y'][sample_index, 0:start, :] = 0
            self.data['Y'][sample_index, end:, :] = 0
            self.data['Y'][sample_index, 0:start, class_list.index('R')] = 1
            self.data['Y'][sample_index, end:, class_list.index('R')] = 1

    def K_fold_dataset_split(self, CV: int = 5):
        """return K datasets for K-Fold cross validation"""
        self.K_Fold_datasets = {f'cv_{i}': {f'{j}': dict() for j in ['train', 'test']} for i in range(1, CV + 1)}

        sample_index_array = np.arange(self.sample_num)
        label_index_array = np.array([[class_index] * self.repeat_num for class_index in range(self.class_num)]).ravel()

        skf = model_selection.StratifiedKFold(n_splits=CV, shuffle=True, random_state=0)
        for index, (train_index, test_index) in enumerate(skf.split(sample_index_array, label_index_array)):
            for stage in ['train', 'test']:
                for item in ['index', 'X', 'Y', 'start', 'end']:
                    if item == 'index':
                        self.K_Fold_datasets[f'cv_{index + 1}'][stage][item] = eval(f'{stage}_index')
                    else:
                        self.K_Fold_datasets[f'cv_{index + 1}'][stage][item] = self.data[item][eval(f'{stage}_index')]

        return self.K_Fold_datasets


class MyModel(object):

    def __init__(self, subj_name: str, RNN1_UnitNum: int = 50, Dense_UnitNum: int = 100, RNN2_UnitNum: int = 50,
                 DropOutRate: float = 0, LearningRate: float = 0.01, class_num: int = 21,
                 timestep_num: int = 400, channel_num: int = 8):
        self.subj_name = subj_name
        self.__RNN1_UnitNum = RNN1_UnitNum
        self.__Dense_UnitNum = Dense_UnitNum
        self.__RNN2_UnitNum = RNN2_UnitNum
        self.__DropOutRate = DropOutRate
        self.class_num = class_num
        self.LearningRate = LearningRate
        self.timestep_num = timestep_num
        self.channel_num = channel_num
        self.model: models.Model

    def initialize_model(self):
        input = layers.Input(shape=(self.timestep_num, self.channel_num))
        L1 = layers.GRU(self.__RNN1_UnitNum, return_sequences=True)(input)
        L2 = layers.Dense(self.__Dense_UnitNum, activation='tanh')(L1)
        L3 = layers.GRU(self.__RNN2_UnitNum, return_sequences=True)(L2)
        output = layers.Dense(self.class_num, activation='softmax')(L3)
        self.model = models.Model(input, output)
        self.model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy(),
                           metrics=[metrics.CategoricalAccuracy()], )

    def train_model(self, train_X, train_Y, test_X, test_Y, model_name, file_save_path,
                    epochs=100, EarlyStop_patience=30,
                    ):
        my_callbacks = [
            callbacks.EarlyStopping(monitor='val_loss', patience=EarlyStop_patience, restore_best_weights=True),
            callbacks.CSVLogger(filename=file_save_path + f'/Log_{model_name}.csv'), ]
        self.model.fit(x=train_X, y=train_Y, epochs=epochs, validation_data=(test_X, test_Y),
                       steps_per_epoch=1, validation_steps=1, shuffle=True, callbacks=my_callbacks)
        self.model.save(filepath=file_save_path + f'/model_{model_name}.h5')

    def load_model(self, model_path):
        self.model = models.load_model(model_path)


if __name__ == "__main__":
    hgp = HGP(subj_name='xx', w1=20, w2=10, start_thresh=8, save_detection_plots=False,)
    hgp.train()
    # hgp.test()
