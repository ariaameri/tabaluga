
def do():

    from deepmlframework.util.config import ConfigParser
    from deepmlframeworksample.trainer.consumer import SampleTrainer

    with open('deepmlframeworksample/config/config.yaml') as f:
        import yaml

        the_json = yaml.full_load(f)
        config = ConfigParser(the_json)

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # config.update({}, {'$set': {'trainer.universal_logger.console': 5}}).print()


    s = SampleTrainer(config.get('trainer'))
    # s = SamplePyTorchTrainer(config.get('trainer'))

    # signal.signal(signal.SIGINT, s.signal_catcher)
    # signal.signal(signal.SIGTERM, s.signal_catcher)
    # signal.signal(signal.SIGTSTP, s.signal_catcher)
    # signal.signal(signal.SIGCONT, s.signal_catcher)

    s.train()




if __name__ == '__main__':

    do()