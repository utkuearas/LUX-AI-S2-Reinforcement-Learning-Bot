from Learner import Learner


if __name__ == "__main__":

    learner = Learner(128,8,"TessEnv-v1",4)

    learner.start_learning()