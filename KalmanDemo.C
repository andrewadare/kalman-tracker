#include "KalFilter.h"
#include "TDecompChol.h"
#include "TRandom3.h"
#include "UtilFns.h"
#include "TSystem.h"
#include <vector>

TMatrixD SampleMVN(TVectorD &mu, TMatrixD &Sigma, int n);

void KalmanDemo()
{
  // State size (number of phase-space coordinates). Here (x, y, xdot, ydot).
  int ss = 4;

  // Observation size (number of observed coordinates). Here (x, y).
  int os = 2;

  // Number of measurement points (radar blips, detector hits, etc.)
  int nsteps = 15;

  // Transition matrix
  TMatrixD F(ss,ss);
  F.UnitMatrix();
  F(0,2) = 1;
  F(1,3) = 1;

  // Observation matrix
  TMatrixD H(os,ss);
  H.UnitMatrix();

  // Transition covariance
  TMatrixD Q(ss,ss);
  Q.UnitMatrix();
  Q *= 0.001;

  // Observation covariance
  TMatrixD R(os,os);
  R.UnitMatrix();
  R *= 1.0;

  // Mu at first point (x,y,xdot,ydot)
  TVectorD mu0(ss);
  mu0(0) = 8; mu0(1) = 10; mu0(2) = 1; mu0(3) = 0;

  // Covariance at first point
  TMatrixD V0(ss,ss);
  V0.UnitMatrix();
  V0 *= 1.0;

  // Setup a tracking demo using the inputs above
  TVectorD ssZero(ss), osZero(os);
  TMatrixD processNoiseSamples = SampleMVN(ssZero, Q, nsteps);
  TMatrixD measureNoiseSamples = SampleMVN(osZero, R, nsteps);
  processNoiseSamples.Print();
  measureNoiseSamples.Print();

  // True (position, velocity) states and observed positions
  TMatrixD trueStates(ss,nsteps);
  TMatrixD measStates(os,nsteps);

  for (int t=0; t<nsteps; t++)
  {
    TVectorD pnoise = TMatrixDRow(processNoiseSamples, t);
    TVectorD mnoise = TMatrixDRow(measureNoiseSamples, t);

    if (t==0)
    {
      // First true state
      TMatrixDColumn(trueStates, t) = mu0;

      // First measured position
      TMatrixDColumn(measStates, t) = H*mu0 + mnoise;
    }
    else
    {
      TVectorD tsprev = TMatrixDColumn(trueStates, t-1);
      TVectorD msprev = TMatrixDColumn(measStates, t-1);

      // Truth and observed states at step t
      TMatrixDColumn(trueStates, t) = F*tsprev + pnoise;
      TMatrixDColumn(measStates, t) = H*tsprev + mnoise;
    }
  }

  trueStates.Print();
  measStates.Print();

  // Forward-only KF
  std::vector<TVectorD> mu(nsteps, mu0);
  std::vector<TMatrixD> V(nsteps, V0);
  KalmanFilter(measStates, F, H, Q, R, mu, V);

  // Forward pass + reverse smoothing pass
  std::vector<TVectorD> mu_smooth(nsteps, mu0);
  std::vector<TMatrixD> V_smooth(nsteps, V0);
  KalmanSmoother(measStates, F, H, Q, R, mu_smooth, V_smooth);

  TGraph *gt = new TGraph(); // True track
  TGraph *gm = new TGraph(); // Noisy observations of track points
  TGraph *gk = new TGraph(); // Filtered track
  TGraph *gs = new TGraph(); // Filtered + smoothed track

  SetGraphProps(gt, kBlack, kNone, kBlack, kOpenSquare);
  SetGraphProps(gm, kBlack, kNone, kGreen+2, kOpenCircle);
  SetGraphProps(gk, kRed, kNone, kRed, kFullCircle);
  SetGraphProps(gs, kBlue, kNone, kBlue, kFullCircle);

  TCanvas c("c", "c", 1);
  TH1F* hf = c.DrawFrame(7,3,23,15);

  for (int t=0; t<nsteps; t++)
  {
    gt->SetPoint(t, trueStates(0,t), trueStates(1,t));
    gm->SetPoint(t, measStates(0,t), measStates(1,t));
    gk->SetPoint(t, mu[t](0), mu[t](1));
    hf->Draw();
    gm->Draw("psame");
    gt->Draw("plsame");
    gk->Draw("plsame");
    gSystem->Sleep(200);
    c.Update();
  }

  for (int t=0; t<nsteps; t++)
  {
    int r = nsteps - t - 1;
    gs->SetPoint(r, mu_smooth[r](0), mu_smooth[r](1));
    gs->Draw("plsame");
    gSystem->Sleep(200);
    c.Update();
  }

  return;
}

TMatrixD SampleMVN(TVectorD &mu, TMatrixD &Sigma, int n)
{
  // Return MVN samples in the rows of a n x d matrix.

  TRandom3 ran;
  ran.SetSeed(0);
  int d = mu.GetNrows();

  // Decompose Sigma = LL'
  TDecompChol chol(Sigma);
  TMatrixD L = chol.GetU();
  L.T();

  TMatrixD Z(d,n);
  for (int i=0; i<d; i++)
    for (int j=0; j<n; j++)
      Z(i,j) = ran.Gaus();

  TMatrixD samples = L*Z;

  // Center at mu
  for (int i=0; i<d; i++)
    for (int j=0; j<n; j++)
      samples(i,j) += mu(i);

  samples.T();
  return samples;
}








