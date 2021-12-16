import librosa
import librosa.display
import numpy as np
from keras.models import load_model
import recording

my_model = load_model('/Users/sumantteegala/OneDrive - University of Massachusetts/Fall_2021/cs_328/328_final_project/Model_mfcc_mean.h5')
path = 'audio_file.wav'
# path = '/Users/sumantteegala/Downloads/Archive/rhappy1.wav'
classes = ["Happy", "Sad"]


def get_feature():
    X, sample_rate = librosa.load(path, res_type='kaiser_fast'
                                      ,duration=2.5
                                      ,sr=44100
                                      ,offset=0.5
                                    )
    sample_rate = np.array(sample_rate)
    test_sample = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    test_sample = np.array(test_sample)
    test_sample = np.mean(test_sample, axis=0)

    # padding with zeros at the end to make the features 216 which will help while using the loaded model to predict
    test_sample = np.pad(test_sample, (0, 216-len(test_sample)), 'constant')
    test_sample = test_sample.reshape((1, 216, 1))

    return test_sample

def onEmotionDetected(emotion):
    print("Speaker-Emotion is {}.".format(emotion))

def predict():
    testing_case = get_feature()

    pred = my_model.predict(testing_case)

    print(pred)
    emotion = pred.argmax(axis=1)[0]

    print(emotion)
    onEmotionDetected(classes[emotion])

    return None

def get_recording():
    recording.record()


if __name__=="__main__":
    get_recording()
    predict()
