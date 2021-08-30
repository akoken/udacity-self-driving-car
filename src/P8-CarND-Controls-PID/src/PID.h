#ifndef PID_H
#define PID_H
#include <vector>

class PID {
   public:
    /**
   * Constructor
   */
    PID();

    /**
   * Destructor.
   */
    virtual ~PID();

    bool twiddle_on;

    /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
    void Init(double Kp_, double Ki_, double Kd_);

    /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
    void UpdateError(double cte);

    /**
   * Calculate the total PID error.
   * @output The total PID error
   */
    double TotalError();

    void Twiddle(double cte);

   private:
    /**
    * PID Errors
    */
    double p_error;
    double i_error;
    double d_error;

    /**
    * PID Coefficients
    */
    double Kp;
    double Ki;
    double Kd;

    int step;
    int batch_size;

    /**
    * Twiddle variables
    */
    std::vector<double> p;
    std::vector<double> best_p;
    std::vector<double> dp;
    double tolarence;
    double best_error;
    double total_error;
    int p_index;
    bool first_run;
    bool second_run;
};

#endif  // PID_H