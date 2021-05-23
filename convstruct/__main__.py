import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from convstruct.libs.util import *
from convstruct.libs.iicc import IICC
from convstruct.libs.core import Core


class Convstruct:
    def __init__(self, compdir=None, indir=None, num_comp=1, num_in=1, num_type="random", name="v1", console=True):
        self.set_args = {'num_comp': num_comp, 'num_in': num_in, 'name': name, 'num_type': num_type, 'indir': indir, 'compdir': compdir, 'console': console}
        self.location = buildLogDir('convstruct', self.set_args['name'])
        self.specifications, self.growth = startSession(self.set_args, self.location)
        self.IICC = IICC(self.set_args, self.location, 'learn')
        self.core = Core(self.set_args, self.location, self.IICC, self.specifications, self.growth)

    def learn(self):
        self.core.runStage(stage=1)

    def live(self):
        while True:
            if self.specifications['live_learning']:
                self.core.runStage(stage=2)
            else:
                break

    def draw(self, stage):
        self.core.runStage(stage)


if __name__ == "__main__":
    args = argsParser()
    Convstruct(compdir=args.compdir, indir=args.indir, num_comp=args.num_comp, num_in=args.num_in, num_type=args.num_type, name=args.name, console=args.console)

tf.logging.set_verbosity(old_v)
