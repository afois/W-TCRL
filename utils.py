import numpy as np
from brian2 import *

def normalized_pos (nb_points):
    resol = 1 / nb_points
    start_val = resol/2
    end_val = 1 - start_val

    norm_pos = np.linspace(start_val, end_val, nb_points)
    return norm_pos

@implementation('cpp', '''
#include<math.h>
double tor_dist(double x1, double y1, double x2, double y2)
{
  double dx = abs(x1 - x2);
  double dy = abs(y1 - y2);

  double tor_dist_x = (dx < 0.5) ? dx : 1.0 - dx;
  double tor_dist_y = (dy < 0.5) ? dy : 1.0 - dy;

 return sqrt(pow(tor_dist_x, 2) + pow(tor_dist_y, 2));
}
''')

@check_units(x1=1, y1=1,x2 = 1, y2 =1,result=1)
def tor_dist(x1, y1, x2, y2):
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    tor_dist_x = np.where(dx < 0.5, dx, 1.0 - dx)
    tor_dist_y = np.where(dy < 0.5, dy, 1.0 - dy)

    return np.sqrt((tor_dist_x**2 +  tor_dist_y**2))

@implementation('cpp', '''
#include<math.h>
double get_lateral_weight(double x1, double y1, double x2, double y2, double sigma, double min_inh, double max_inh)
{
  double dx = abs(x1 - x2);
  double dy = abs(y1 - y2);

  double tor_dist_x = (dx < 0.5) ? dx : 1.0 - dx;
  double tor_dist_y = (dy < 0.5) ? dy : 1.0 - dy;

  double dist = sqrt(pow(tor_dist_x, 2) + pow(tor_dist_y, 2));
  double res = 0;

  if (dist <= sigma)
  {
      res = min_inh * dist;
  }

  else
  {
    res = max_inh;
  }

   return res;
}
''')

@check_units(x1=1, y1=1,x2 = 1, y2 =1, sigma = 1, min_inh = 1, max_inh = 1, result=1)
def get_lateral_weight(x1, y1, x2, y2, sigma, min_inh, max_inh):
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    tor_dist_x = np.where(dx < 0.5, dx, 1.0 - dx)
    tor_dist_y = np.where(dy < 0.5, dy, 1.0 - dy)

    dist = np.sqrt((tor_dist_x**2 +  tor_dist_y**2))
    res = 0
    if (dist <= sigma):
        res = min_inh * dist
    else :
        res = max_inh

    return res
