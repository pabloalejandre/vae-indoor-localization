from utils.visualize_utils import  read_point_coordinates, draw_points_in_room

scenario_name, width, length = 'BoxLectureRoom', 10, 19
#scenario_name, width, length = 'LShapedRoom', 10, 19
#scenario_name, width, length = 'LivingRoom', 7, 7
csv_path = 'scenarios/' + scenario_name + '/csv/'

training = read_point_coordinates(csv_path + 'training.csv')
transmitter = read_point_coordinates(csv_path + 'transmitter.csv')
fingerprints = read_point_coordinates(csv_path + 'fingerprints.csv')
testing_points = read_point_coordinates(csv_path + 'testing_points.csv')

draw_points_in_room(width, length, transmitter, training, testing_points, show_index=False, LRoom=False, draw_grid=False)
