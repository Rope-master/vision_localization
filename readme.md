## Usage

```.sh
git clone git@github.com:trunktech/vision_localization.git

mkdir -p catkin_ws/src
cd catkin_ws/src
ln -s ../../vision_localization .
cd ..
catkin_make

roscore

# then open a new terminal
. devel/setup.bash
roslaunch vision_localization detection.launch
```

## Configuration

Before running `roslaunch`, you have to configure the node so
that it knows where to find the camera parameter file and
the list of images.

Change the file `config/default.yaml`, which reads

```yaml
camera_parameter_filename: /home/fangjun/task1/vision_localization/param/pudong.yaml
image_list_filename: /media/hebei/787C78777C7831CC/datasets/shanghai201711/img.list
image_dir: /media/hebei/787C78777C7831CC/datasets/shanghai201711/2017-11-13-11-35-53
```

The file for `camera_parameter_filename` should have the following format

```yaml
{
  fx: 700.0,
  fy: 700.0,
  cx: 646.0,
  cy: 482.0,
  pitch: 10.3,
  yaw: 0.0,
  roll: 0.0,
  in_x1: 306,
  in_x2: 986,
  in_y1: 462,
  in_y2: 964,
  out_w: 448,
  out_h: 448
}
```

The file for `image_list_filename` is just a text file
with an arbitrary extension and each line contains
an image filename **relative** to `image_dir`.

## Test

To run the test, execute

```.sh
catkin_make tests
./devel/lib/vision_localization/vision_localization-test
```

## Documentation

To generate Doxygen style documentation you have to install
Doxygen via

```.sh
sudo apt-get install doxygen graphviz graphviz-dev
```

and then run

```.sh
cd doc
./run-doxygen.sh
```

You can find the generated documentation inside `doc/html`.
