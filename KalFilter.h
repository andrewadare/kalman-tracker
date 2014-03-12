#include "TMatrixD.h"
#include "TVectorD.h"
#include <vector>

void KalmanFilter(TMatrixD &coords,  // Observed positions (columns) in t steps
                  TMatrixD &A,
                  TMatrixD &C,
                  TMatrixD &Q,
                  TMatrixD &R,
                  std::vector<TVectorD> &mu,
                  std::vector<TMatrixD> &V)
{
  int nsteps = coords.GetNcols();

  // For calculations below
  TMatrixD I(V[0]); I.UnitMatrix();
  TMatrixD AT(A); AT.T();
  TMatrixD CT(C); CT.T();

  for (int t=0; t<nsteps; t++)
  {
    // Predicted mean and covariance
    TVectorD predmu = t ? A*mu[t-1] : mu[0];
    TMatrixD predV  = t ? A*V[t-1]*AT + Q : V[0];

    // Error (innovation) vector
    TVectorD obs = TMatrixDColumn(coords,t);
    TVectorD e = obs - C*predmu;

    // Kalman gain matrix K
    TMatrixD S = C*predV*CT + R;
    S.Invert();
    TMatrixD K = predV*CT*S;

    // Output: mean and covariance of next state.
    if (t)
    {
      mu[t] = predmu + K*e;
      V[t]  = (I - K*C)*predV;
    }
  }

  return;
}

void KalmanSmoother(TMatrixD &coords,
                    TMatrixD &A,
                    TMatrixD &C,
                    TMatrixD &Q,
                    TMatrixD &R,
                    std::vector<TVectorD> &mu,
                    std::vector<TMatrixD> &V)
{
  int nsteps = coords.GetNcols();

  // Forward pass
  KalmanFilter(coords, A, C, Q, R, mu, V);

  // For calculations below
  TMatrixD AT(A); AT.T();
  TMatrixD CT(C); CT.T();

  for (int t=nsteps-2; t>=0; t--)
  {
    // Predicted mean and covariance
    TVectorD predmu = A*mu[t];          // mu(t+1|t)
    TMatrixD predV  = A*V[t]*AT + Q;    // cov(t+1|t)
    TMatrixD predVinv(predV);
    predVinv.Invert();

    // Smoother gain matrix
    TMatrixD J = V[t] * AT * predVinv;
    TMatrixD JT(J); J.T();

    // Update mu and V
    mu[t] += J*(mu[t+1] - predmu);
    V[t]  += J*(V[t+1]  - predV)*JT;
  }

  return;

}

















