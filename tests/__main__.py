import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from convstruct.libs.art import *
import shutil
import unittest

# -----To repress Tensorflow deprecation warnings----- #
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Test(unittest.TestCase):
    def test_a_build_log_dir(self):
        if os.path.exists(os.path.join('tests', 'void')):
            shutil.rmtree(os.path.join('tests', 'void'))
        if os.path.exists(os.path.join('tests', 'test_void')):
            shutil.rmtree(os.path.join('tests', 'test_void'))
        if os.path.exists(os.path.join('data', 'test_comp_data')):
            shutil.rmtree(os.path.join('data', 'test_comp_data'))
        if os.path.exists(os.path.join('data', 'test_input_data')):
            shutil.rmtree(os.path.join('data', 'test_input_data'))
        build_log_dir('tests', 'void')
        build_log_dir('data', 'test_input_data')
        build_log_dir('data', 'test_comp_data')
        self.assertTrue(path.exists('tests/void'))
        self.assertTrue(path.exists('data/test_input_data'))
        self.assertTrue(path.exists('data/test_comp_data'))

    def test_b_create_log(self):
        create_log(os.path.join('tests', 'void'), 'unittest.log')
        self.assertTrue(path.exists('tests/void/unittest.log'))

    def test_c_prepare_image(self):
        created_image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        created_nonint_image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        created_small_image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        created_reshape_input_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        created_reshape_output_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        created_prime_input_image = Image.fromarray((np.random.rand(129, 129, 3) * 255).astype(np.uint8))
        created_prime_output_image = Image.fromarray((np.random.rand(129, 129, 3) * 255).astype(np.uint8))
        created_image.save('data/test_comp_data/test_output.png')
        created_image.save('data/test_input_data/test_input.png')
        created_image.save('data/test_comp_data/test_output.jpeg')
        created_image.save('data/test_comp_data/test_output.bmp')
        created_nonint_image.save('data/test_comp_data/test_nonint_output.png')
        created_small_image.save('data/test_input_data/test_small_input.png')
        created_small_image.save('data/test_comp_data/test_small_output.png')
        created_prime_input_image.save('data/test_input_data/test_prime_input.png')
        created_prime_output_image.save('data/test_comp_data/test_prime_output.png')
        created_reshape_input_image.save('data/test_input_data/test_reshape_input.png')
        created_reshape_output_image.save('data/test_comp_data/test_reshape_output.png')
        self.assertTrue(path.exists('data/test_comp_data/test_output.png'))
        self.assertTrue(path.exists('data/test_input_data/test_input.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_nonint_output.png'))
        self.assertTrue(path.exists('data/test_input_data/test_small_input.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_small_output.png'))
        self.assertTrue(path.exists('data/test_input_data/test_prime_input.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_prime_output.png'))
        self.assertTrue(path.exists('data/test_input_data/test_reshape_input.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_reshape_output.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_output.jpeg'))
        self.assertTrue(path.exists('data/test_comp_data/test_output.bmp'))

    def test_d_process_image(self):
        img_png = process_image(100, 2, 'data/test_comp_data/test_output.png', 64, 64)
        img_jpg = process_image(100, 2, 'data/test_comp_data/test_output.jpeg', 64, 64)
        img_bmp = process_image(100, 2, 'data/test_comp_data/test_output.bmp', 64, 64)
        sess = tf.Session(config=tf_config_setup())
        res_png, res_jpg, res_bmp = sess.run([img_png, img_jpg, img_bmp])
        Image.fromarray(res_png.astype(np.uint8)).save('data/test_comp_data/test_result_output.png')
        Image.fromarray(res_jpg.astype(np.uint8)).save('data/test_comp_data/test_result_output.jpeg')
        Image.fromarray(res_bmp.astype(np.uint8)).save('data/test_comp_data/test_result_output.bmp')
        self.assertTrue(path.exists('data/test_comp_data/test_result_output.png'))
        self.assertTrue(path.exists('data/test_comp_data/test_result_output.jpeg'))
        self.assertTrue(path.exists('data/test_comp_data/test_result_output.bmp'))

    def test_e_convstruct_start(self):
        # To test starting points use this in indir: 'data/test_input_data'
        set_args = {'num_comp': 1, 'num_in': 1, 'name': "void", 'num_type': "random", 'indir': None, 'compdir': 'data/test_comp_data', 'console': True}
        location = build_log_dir('tests', set_args['name'])
        specifications, growth = convstruct_start(set_args, location)
        specifications['total_epoch_count'], specifications['max_filter_size'], specifications['max_kernel_size'], specifications['max_stride_size'], specifications['num_gpus'] = 10, 32, 3, 2, 1
        growth['initial_length'], growth['estimator_length'], growth['iicc_length'], growth['full_length'], growth['model_target'] = 10, 100, 50, 50, 0
        learn = IICC(set_args, location, 'learn')
        building = Building(set_args, location, learn, specifications, growth)
        return set_args, location, specifications, growth, building

    def test_f_convstruct_learn(self):
        tf.reset_default_graph()
        args, location, specifications, growth, building = self.test_e_convstruct_start()
        building.start(stage=1)
        return self.assertTrue(os.path.exists('tests/void/iicc/learn.ckpt.meta'))

    def test_g_convstruct_live(self):
        args, location, specifications, growth, building = self.test_e_convstruct_start()
        if specifications['gpus'] < specifications['num_gpus']:
            specifications['gpus'] = specifications['num_gpus']
            specifications['multi_gpu_test'] = True
            for batch in range(specifications['gpus']):
                specifications['max_memory_%d' % batch] = specifications['max_memory_0']
        while True:
            if specifications['live_learning']:
                building.start(stage=2)
            else:
                break
        return self.assertTrue(os.path.exists('tests/void/live.ckpt.meta'))

    def test_h_convstruct_draw_a(self):
        args, location, specifications, growth, building = self.test_e_convstruct_start()
        building.start(stage=3)
        self.assertTrue(not growth['draw_learning'])
        self.assertTrue(growth['saved_epoch'] != 0)
        return self.assertTrue(os.path.exists('tests/void/draw/draw.ckpt.meta'))

    def test_i_convstruct_draw_b(self):
        tf.reset_default_graph()
        args, location, specifications, growth, building = self.test_e_convstruct_start()
        building.start(stage=4)
        self.assertTrue(path.exists('tests/void/draw/final_output_1_1.png'))


if __name__ == '__main__':
    unittest.main()
