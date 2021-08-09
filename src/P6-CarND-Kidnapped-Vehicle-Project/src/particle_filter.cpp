/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */

  std::default_random_engine gen;
  double std_x, std_y, std_theta;

  //Set standard deviations for x, y, and theta
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);

  // TODO: Set the number of particles
  //num_particles = 1000; gives out of time error
  num_particles = 500;

  for (unsigned int i = 0; i < num_particles; ++i) {
    //Create a new particle
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    std::default_random_engine gen;
    for (unsigned int i = 0; i < num_particles; ++i) {
      double theta_old = particles[i].theta;

      if(fabs(yaw_rate) < 0.000001){
        particles[i].x += velocity*delta_t*cos(theta_old);
        particles[i].y += velocity*delta_t*sin(theta_old);
      }else {
        particles[i].theta += yaw_rate*delta_t;
        particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta) - sin(theta_old));
        particles[i].y += (velocity/yaw_rate)*(cos(theta_old) - cos(particles[i].theta));
      }

      std::normal_distribution<double> dist_x(0, std_pos[0]);
      std::normal_distribution<double> dist_y(0, std_pos[1]);
      std::normal_distribution<double> dist_theta(0, std_pos[2]);

      particles[i].x += dist_x(gen);
      particles[i].y += dist_y(gen);
      particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); ++i) {
      int obs_id = 0;
	    LandmarkObs obs = observations[i];
	    double min_dist = std::numeric_limits<double>::max();

	    for (unsigned int j = 0; j < predicted.size(); ++j) {
	      LandmarkObs pred = predicted[j];
	      double dist_curr = dist(obs.x, obs.y, pred.x, pred.y);
	      if (dist_curr < min_dist) {
	        min_dist = dist_curr;
          obs_id = pred.id;
	      }
	    }
	    observations[i].id = obs_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (unsigned int i = 0; i < num_particles; i++) {
    double p_theta = particles[i].theta;
    double p_x = particles[i].x;
    double p_y = particles[i].y;

    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int l_id = map_landmarks.landmark_list[j].id_i;
      float l_x = map_landmarks.landmark_list[j].x_f;
      float l_y = map_landmarks.landmark_list[j].y_f;

      if (fabs(l_x - p_x) <= sensor_range && fabs(l_y - p_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{ l_id, l_x, l_y });
      }
    }

    // Transform vehicle coordinates to map coordinates.
    vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double transformed_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double transformed_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
    }

    dataAssociation(predictions, transformed_obs);
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      double o_x, o_y, pr_x, pr_y;
      o_x = transformed_obs[j].x;
      o_y = transformed_obs[j].y;

      int associated_pred = transformed_obs[j].id;
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_pred) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }

      double s_x = std_landmark[0];
      double s_y = std_landmark[1];

      //Calculate multivariate Gaussian
      double obs_weight = (1/(2*M_PI*s_x*s_y)) * exp(-(pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2)))));
      particles[i].weight *= obs_weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::vector<double> weights(particles.size());
  std::vector<Particle> resampled_particles(num_particles);

  for (unsigned int i = 0; i < particles.size(); i++){
    weights[i] = particles[i].weight;
  }

  std::default_random_engine gen;
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  for (unsigned int j = 0; j < particles.size(); j++) {
    resampled_particles[j] = particles[dist(gen)];
  }

	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}