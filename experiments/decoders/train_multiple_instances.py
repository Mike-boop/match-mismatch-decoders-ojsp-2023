from experiments.decoders.decoders import DilatedConvNet, DilatedConvNetSymmetrisedOutputs
from experiments.decoders.train_instance import train_instance


if __name__ == '__main__':

    config_path = '/home/mdt20/Code/match-mismatch-decoders-ojsp-2023/config.json'
    model = DilatedConvNetSymmetrisedOutputs()

    response = 'env'
    use_baseline = False
    eval_only = False

    for inst in range(100):

        # baseline model
        train_instance(config_path,
                       DilatedConvNet(),
                       inst=inst,
                       response='env')
        
        # envelope-based model
        train_instance(config_path,
                       DilatedConvNetSymmetrisedOutputs(),
                       inst=inst,
                       response='env')
        
        # ffr-based model
        train_instance(config_path,
                       DilatedConvNetSymmetrisedOutputs(),
                       inst=inst,
                       response='ffr')