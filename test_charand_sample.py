from nexdeepml.util.config import ConfigParser
# from nexdeepmlsample.trainer.consumer import SampleTrainer
from nexdeepmlsample.trainer.consumer import SamplePyTorchTrainer
# import signal


if __name__ == '__main__':

    with open('nexdeepml/config/config.yaml') as f:
        import yaml

        the_json = yaml.full_load(f)
        config = ConfigParser(the_json)

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # s = SampleTrainer(config.get('trainer'))
    s = SamplePyTorchTrainer(config.get('trainer'))

    # signal.signal(signal.SIGINT, s.signal_catcher)
    # signal.signal(signal.SIGTERM, s.signal_catcher)
    # signal.signal(signal.SIGTSTP, s.signal_catcher)
    # signal.signal(signal.SIGCONT, s.signal_catcher)

    s.train()
