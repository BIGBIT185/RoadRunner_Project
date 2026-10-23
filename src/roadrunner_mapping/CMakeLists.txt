cmake_minimum_required(VERSION 3.0.2)
project(roadrunner_mapping)
find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs grid_map_msgs message_filters message_generation rospy
  sensor_msgs std_msgs std_srvs tf2_ros)
catkin_python_setup()
add_message_files(FILES EomtSemanticPrediction.msg)
generate_messages(DEPENDENCIES sensor_msgs std_msgs)
catkin_package(CATKIN_DEPENDS diagnostic_msgs grid_map_msgs message_filters
  message_runtime rospy sensor_msgs std_msgs std_srvs tf2_ros)
catkin_install_python(PROGRAMS scripts/mapping_node.py scripts/semantic_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS scripts/run_semantic.bash DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
