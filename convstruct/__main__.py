import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from convstruct.libs.art import *


class Convstruct:
    def __init__(self, compdir=None, indir=None, num_comp=1, num_in=1, num_type="random", name="v1", console=True):
        self.set_args = {'num_comp': num_comp, 'num_in': num_in, 'name': name, 'num_type': num_type, 'indir': indir, 'compdir': compdir, 'console': console}
        self.location = build_log_dir('convstruct', self.set_args['name'])
        self.specifications, self.growth = convstruct_start(self.set_args, self.location)
        self.IICC = IICC(self.set_args, self.location, 'learn')
        self.building = Building(self.set_args, self.location, self.IICC, self.specifications, self.growth)

    def learn(self):
        self.building.start(stage=1)

    def live(self):
        while True:
            if self.specifications['live_learning']:
                self.building.start(stage=2)
            else:
                break

    def draw(self, stage):
        self.building.start(stage)


if __name__ == "__main__":
    args = args_parser()
    Convstruct(compdir=args.compdir, indir=args.indir, num_comp=args.num_comp, num_in=args.num_in, num_type=args.num_type, name=args.name, console=args.console)

tf.logging.set_verbosity(old_v)
