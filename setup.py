import os # Operating system library
from glob import glob # Handles file path names
from setuptools import setup # Facilitates the building of packages

package_name = 'rl_bot'

# Path of the current directory
cur_directory_path = os.path.abspath(os.path.dirname(__file__))


def get_all_files_recursively(folder):
    return [f for f in glob(os.path.join(folder, '**'), recursive=True) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Path to the launch file      
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),

        # Path to the config file
        (os.path.join('share', package_name,'config'), glob('config/*.yaml')),

        # Path to the world file
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),
        
        # Path to the urdf file
        (os.path.join('share', package_name,'urdf/'), glob('./urdf/*')),

        # Path to the meshes file
        # (os.path.join('share', package_name,'meshes/'), get_all_files_recursively('meshes')),
        
        # # Path to the mobile robot sdf and config file
        # (os.path.join('share', package_name,'models/mobile_warehouse_robot/'), glob('./models/mobile_warehouse_robot/*')),
        
        # Path to the pioneer sdf file
        (os.path.join('share', package_name,'models/turtlebot3_waffle/'), glob('./models/turtlebot3_waffle/model.sdf')),

        # Path to the pioneer config file
        (os.path.join('share', package_name,'models/turtlebot3_waffle/'), glob('./models/turtlebot3_waffle/model.config')),

        # Path to the meshes file
        (os.path.join('share', package_name,'models/turtlebot3_waffle/meshes/'), glob('./models/turtlebot3_waffle/meshes/*')),
        
        # Path to the target sdf file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.sdf')),

        # Path to the target config file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.config')),

        # Path to the world file (i.e. warehouse + global environment)
        (os.path.join('share', package_name,'models/'), glob('./worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhinav Bhamidipati',
    maintainer_email='abhinav7@terpmail.umd.edu',
    description='This package contains the RL bot and its environment',
    license='use everywhere and anywhere',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'spawn_demo = rl_bot.spawn_demo:main',
          'start_training = rl_bot.start_training:main',
          'trained_agent = rl_bot.trained_agent:main',
          'start_training_SAC = rl_bot.start_training_SAC:main',
          'trained_agent_SAC = rl_bot.trained_agent_SAC:main',
        ],
    },
)