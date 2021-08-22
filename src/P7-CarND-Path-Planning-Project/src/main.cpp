#include <uWS/uWS.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "constants.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
    uWS::Hub h;

    // Load up map values for waypoint's x,y,s and d normalized normal vectors
    vector<double> map_waypoints_x;
    vector<double> map_waypoints_y;
    vector<double> map_waypoints_s;
    vector<double> map_waypoints_dx;
    vector<double> map_waypoints_dy;

    // Waypoint map to read from
    string map_file_ = "../data/highway_map.csv";
    // The max s value before wrapping around the track back to 0
    double max_s = 6945.554;

    std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        std::istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }

    int lane = 1;
    double ref_vel = 0;

    h.onMessage([&ref_vel, &map_waypoints_x, &map_waypoints_y, &map_waypoints_s, &map_waypoints_dx,
                 &map_waypoints_dy, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        if (length && length > 2 && data[0] == '4' && data[1] == '2') {
            auto s = hasData(data);

            if (s != "") {
                auto j = json::parse(s);

                string event = j[0].get<string>();

                if (event == "telemetry") {
                    // j[1] is the data JSON object

                    // Main car's localization Data
                    double car_x = j[1]["x"];
                    double car_y = j[1]["y"];
                    double car_s = j[1]["s"];
                    double car_d = j[1]["d"];
                    double car_yaw = j[1]["yaw"];
                    double car_speed = j[1]["speed"];

                    // Previous path data given to the Planner
                    auto previous_path_x = j[1]["previous_path_x"];
                    auto previous_path_y = j[1]["previous_path_y"];
                    // Previous path's end s and d values
                    double end_path_s = j[1]["end_path_s"];
                    double end_path_d = j[1]["end_path_d"];

                    // Sensor Fusion Data, a list of all other cars on the same side
                    //   of the road.
                    auto sensor_fusion = j[1]["sensor_fusion"];

                    // Fetch number of points in previous path
                    int prev_path_size = previous_path_x.size();

                    // set current s to last path s if we travelled
                    if (prev_path_size > 0) {
                        car_s = end_path_s;
                    }

                    bool other_vehicle_front = false;
                    bool other_vehicle_left = false;
                    bool other_vehicle_right = false;

                    for (int i = 0; i < sensor_fusion.size(); i++) {
                        float d = sensor_fusion[i][6];
                        // find target vehicle's lane
                        int other_vehicle_lane = FindLane(d);
                        if (other_vehicle_lane < 0) {
                            continue;
                        }

                        // Calculate target vehicle speed
                        double vx = sensor_fusion[i][3];
                        double vy = sensor_fusion[i][4];
                        double other_vehicle_speed = sqrt(vx * vx + vy * vy);
                        double other_vehicle_s = sensor_fusion[i][5];

                        // Calculate target vehicle s distance
                        other_vehicle_s += ((double)prev_path_size * 0.02 * other_vehicle_speed);

                        // Check if there are vehicles in front, left or right of ego vehicle.
                        if (other_vehicle_lane == lane) {
                            if ((other_vehicle_s > car_s) && (other_vehicle_s - car_s) < 30) {
                                other_vehicle_front = true;
                            }
                        } else if (other_vehicle_lane - lane == -1) {
                            if (((car_s - 30) < other_vehicle_s) && ((car_s + 30) > other_vehicle_s)) {
                                other_vehicle_left = true;
                            }
                        } else if (other_vehicle_lane - lane == 1) {
                            if (((car_s - 30) < other_vehicle_s) && ((car_s + 30) > other_vehicle_s)) {
                                other_vehicle_right = true;
                            }
                        }
                    }

                    double delta_v = 0;
                    if (other_vehicle_front) {
                        // check if it's safe to change lane left.
                        if (!other_vehicle_left && lane > 0) {
                            lane--;
                        } else if (!other_vehicle_right && lane != 2) {
                            lane++;
                        } else
                        {
                            delta_v -= MAX_ACC;//It's not safe to change lane, so slow down.
                        }
                    } else {
                        if (lane != 1)  // check if ego vehicle is in the middle lane.
                        {
                            if ((lane == 0 && !other_vehicle_right) || (lane == 2 && !other_vehicle_left)) {
                                lane = 1;  // change lane to middle
                            }
                        }
                        // if ego vehicle below max speed limit, accelerate.
                        if (ref_vel < MAX_SPEED_LIMIT) {
                            delta_v += MAX_ACC;
                        }
                    }

                    json msgJson;

                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    vector<double> pts_x;
                    vector<double> pts_y;

                    double ref_ego_x = car_x;
                    double ref_ego_y = car_y;
                    double ref_ego_yaw = deg2rad(car_yaw);

                    if (prev_path_size >= 2) {
                        ref_ego_x = previous_path_x[prev_path_size - 1];
                        ref_ego_y = previous_path_y[prev_path_size - 1];

                        double prev_ref_ego_x = previous_path_x[prev_path_size - 2];
                        double prev_ref_ego_y = previous_path_y[prev_path_size - 2];

                        pts_x.push_back(prev_ref_ego_x);
                        pts_x.push_back(ref_ego_x);
                        pts_y.push_back(prev_ref_ego_y);
                        pts_y.push_back(ref_ego_y);

                        ref_ego_yaw = atan2(ref_ego_y - prev_ref_ego_y, ref_ego_x - prev_ref_ego_x);
                    } else {
                        double prev_car_x = car_x - cos(ref_ego_yaw);
                        double prev_car_y = car_y - sin(ref_ego_yaw);

                        pts_x.push_back(prev_car_x);
                        pts_x.push_back(car_x);
                        pts_y.push_back(prev_car_y);
                        pts_y.push_back(car_y);
                    }

                    vector<double> waypoint0 = getXY(car_s + 30, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    vector<double> waypoint1 = getXY(car_s + 60, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    vector<double> waypoint2 = getXY(car_s + 90, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

                    pts_x.push_back(waypoint0[0]);
                    pts_x.push_back(waypoint1[0]);
                    pts_x.push_back(waypoint2[0]);

                    pts_y.push_back(waypoint0[1]);
                    pts_y.push_back(waypoint1[1]);
                    pts_y.push_back(waypoint2[1]);

                    // transform coordinates to car's reference
                    for (int i = 0; i < pts_x.size(); i++) {
                        double delta_x = pts_x[i] - ref_ego_x;
                        double delta_y = pts_y[i] - ref_ego_y;

                        pts_x[i] = delta_x * cos(0 - ref_ego_yaw) - delta_y * sin(0 - ref_ego_yaw);
                        pts_y[i] = delta_x * sin(0 - ref_ego_yaw) + delta_y * cos(0 - ref_ego_yaw);
                    }

                    tk::spline sp;
                    sp.set_points(pts_x, pts_y);

                    for (int i = 0; i < prev_path_size; i++) {
                        next_x_vals.push_back(previous_path_x[i]);
                        next_y_vals.push_back(previous_path_y[i]);
                    }

                    double target_x = 30;
                    double target_y = sp(target_x);
                    double target_distance = sqrt(target_x * target_x + target_y * target_y);

                    double x_cumulative = 0;

                    for (int i = 1; i <= 50 - prev_path_size; i++) {
                        ref_vel += delta_v;
                        if (ref_vel > MAX_SPEED_LIMIT) {
                            ref_vel = MAX_SPEED_LIMIT;
                        } else if (ref_vel < MAX_ACC) {
                            ref_vel = MAX_ACC;
                        }
                        // number of points
                        double N = (target_x / (0.02 * ref_vel / 2.24));

                        // point coordinates in car's frame
                        double x_car_frame = x_cumulative + target_x / N;
                        double y_car_frame = sp(x_car_frame);

                        x_cumulative = x_car_frame;

                        double x_current = ref_ego_x + x_car_frame * cos(ref_ego_yaw) - y_car_frame * sin(ref_ego_yaw);
                        double y_current = ref_ego_y + x_car_frame * sin(ref_ego_yaw) + y_car_frame * cos(ref_ego_yaw);

                        next_x_vals.push_back(x_current);
                        next_y_vals.push_back(y_current);
                    }

                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"control\"," + msgJson.dump() + "]";

                    //this_thread::sleep_for(chrono::milliseconds(1000));
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }  // end websocket if
    });    // end h.onMessage

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                           char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }

    h.run();
}