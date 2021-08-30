#include "PID.h"

#include <iostream>
#include <limits>
#include <vector>

#include "math.h"

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
    p = {Kp_, Ki_, Kd_};
    best_p = p;
    dp = {0.1 * Kp, 0.1 * Ki, 0.1 * Kd};

    p_error = 0;
    i_error = 0;
    d_error = 0;

    //Twiddle parameters
    step = 0;
    batch_size = 100;  //Number of iterations to average the best parameters
    best_error = std::numeric_limits<double>::max();
    total_error = 0;  // Keeps track of total error in a batch
    tolarence = 0.00001;
    p_index = 0;  //Parameter index
    first_run = true;
    second_run = true;
}

void PID::UpdateError(double cte) {
    d_error = cte - p_error;
    p_error = cte;
    i_error += cte;

    step += 1;

    if (twiddle_on) {
        Twiddle(cte);
    }
}

double PID::TotalError() {
    double total_error = -1 * (p[0] * p_error + p[1] * i_error + p[2] * d_error);
    return total_error;
}

void PID::Twiddle(double cte) {
    total_error += fabs(cte);

    //Calculate average CTE
    if (step > batch_size) {
        total_error /= batch_size;

        //first run through the twiddle algorithm
        if (first_run) {
            // Increase the parameter
            p[p_index] += dp[p_index];
            first_run = false;
        } else {
            if (total_error <= best_error && second_run) {
                //if increasing improved the error, keep increasing
                best_error = total_error;
                best_p[p_index] = p[p_index];
                dp[p_index] *= 1.1;
                p[p_index] += dp[p_index];
            } else {
                if (second_run) {
                    //if increasing did not improve error, decrease
                    p[p_index] -= 2 * dp[p_index];
                    second_run = false;
                } else {
                    if (total_error < best_error) {
                        best_error = total_error;
                        best_p[p_index] = p[p_index];
                        dp[p_index] *= 1.1;
                    } else {
                        p[p_index] += dp[p_index];
                        dp[p_index] *= 0.8;
                    }

                    // Next parameter
                    p_index = (p_index + 1) % 3;

                    //reset for next parameter
                    first_run = true;
                    second_run = true;
                }
            }
        }

        //reset for the next batch
        step = 0;
        total_error = 0;

        //Turn off twiddle if sum(dp) < tolarence
        double sum_dp = fabs(dp[0]) + fabs(dp[1]) + fabs(dp[2]);
        if (sum_dp < tolarence) {
            twiddle_on = false;
            std::cout << "Twiddle Completed!" << std::endl;
            std::cout << "Best Parameters: => " << best_p[0] << ", " << best_p[1] << ", " << best_p[2] << std::endl;
        }

        //std::cout << "Sum(dp): " << sum_dp << std::endl;
        //std::cout << "Best error: " << best_error << std::endl;
        //std::cout << "Best Parameters: => " << best_p[0] << ", " << best_p[1] << ", " << best_p[2] << std::endl;
    }
}