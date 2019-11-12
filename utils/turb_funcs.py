
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import h5py

import skimage.measure
from scipy.stats import kde

def int_shape(x):
  return list(map(int, x.get_shape()))

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def _simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
  return y

def vel_to_norm(vel):
  return np.sqrt(np.sum(np.square(vel), axis=-1))

def energy_spectrum_np(vel):
  """ Compute the energy spectrum of a velocity vector field:
  Args:
    vel: NP Tensor of shape [H, W, 2] or [H, W, D, 3]
  Returns:
    E: a list of energy spectrums for each dim. Either (Eu, Ev) or (Eu, Ev, Ew)
  """

  # get shapes
  nx = vel.shape[:-1]
  nxc = [x/2 + 1 for x in nx]

  # get dim
  dim = vel.shape[-1]

  if dim == 2:
    kmax = int(round(np.sqrt((1-nxc[0])**2 + (1-nxc[1])**2)) +1 )
    Eu = np.zeros(kmax)
    Ev = np.zeros(kmax)

    grid_x = np.linspace(0, nx[0]-1, nx[0])
    grid_y = np.linspace(0, nx[1]-1, nx[1])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_x = grid_x - nxc[0] + 1
    grid_y = grid_y - nxc[1] + 1
    km = np.round(np.sqrt(grid_x**2.0 + grid_y**2.0)).astype(np.int)

    P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,0]))), 2.0)
    for i in range(0, nx[0]):
      for j in range(0, nx[1]):
        Eu[km[i,j]] = Eu[km[i,j]] + P[i,j]
  
    P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,1]))), 2.0)
    for i in range(0, nx[0]):
      for j in range(0, nx[1]):
          Ev[km[i,j]] = Ev[km[i,j]] + P[i,j]

    return Eu, Ev

  elif dim == 3:
    kmax = int(round(np.sqrt((1-nxc[0])**2 + (1-nxc[1])**2 + (1-nxc[2])**2)) + 1)
    Eu = np.zeros(kmax)
    Ev = np.zeros(kmax)
    Ew = np.zeros(kmax)

    grid_x = np.linspace(0, nx[0]-1, nx[0])
    grid_y = np.linspace(0, nx[1]-1, nx[1])
    grid_z = np.linspace(0, nx[2]-1, nx[2])
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    grid_x = grid_x - nxc[0] + 1
    grid_y = grid_y - nxc[1] + 1
    grid_z = grid_z - nxc[2] + 1
    km = np.round(np.sqrt(grid_x**2.0 + grid_y**2.0 + grid_z**2.0)).astype(np.int)

    P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,0]))), 2.0)
    for i in range(0, nx[0]):
      for j in range(0, nx[1]):
        for k in range(0, nx[2]):
          #km = int(round(np.sqrt((i - nxc[0] + 1)**2 + (j - nxc[1] + 1)**2 + (k - nxc[2] + 1)**2)))
          #Eu[km] = Eu[km] + P[i,j,k]
          Eu[km[i,j,k]] = Eu[km[i,j,k]] + P[i,j,k]
  
    P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,1]))), 2.0)
    for i in range(0, nx[0]):
      for j in range(0, nx[1]):
        for k in range(0, nx[2]):
          #km = int(round(np.sqrt((i - nxc[0] + 1)**2 + (j - nxc[1] + 1)**2 + (k - nxc[2] + 1)**2)))
          #Ev[km] = Ev[km] + P[i,j,k]
          Ev[km[i,j,k]] = Ev[km[i,j,k]] + P[i,j,k]
  
    P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,2]))), 2.0)
    for i in range(0, nx[0]):
      for j in range(0, nx[1]):
        for k in range(0, nx[2]):
          #km = int(round(np.sqrt((i - nxc[0] + 1)**2 + (j - nxc[1] + 1)**2 + (k - nxc[2] + 1)**2)))
          #Ew[km] = Ew[km] + P[i,j,k]
          Ew[km[i,j,k]] = Ew[km[i,j,k]] + P[i,j,k]

    return Eu, Ev, Ew

  else:
    print("dimenstion of velocity field not supported")
    print("dim: " + str(dim))
    exit()

def energy_spectrum_tf(vel):
  """ Compute the energy spectrum of a velocity vector field:
  Args:
    vel: TF Tensor of shape [H, W, 2] or [H, W, D, 3]
  Returns:
    E: a list of energy spectrums for each dim. Either (Eu, Ev) or (Eu, Ev, Ew)
  """
  #TODO add batch support (difficult with tf.spectral.rfft functions)

  # get shapes
  nx = int_shape(vel)[:-1]
  nxc = [x/2 + 1 for x in nx]

  # get dim
  dim = int_shape(vel)[-1]
  
  if dim == 2:
    kmax = int(round(np.sqrt((1-nxc[0])**2 + (1-nxc[1])**2)) +1 )
    Eu = np.zeros(kmax)
    Ev = np.zeros(kmax)

    grid_x = np.linspace(0, nx[0]-1, nx[0])
    grid_y = np.linspace(0, nx[1]-1, nx[1])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_x = np.minimum(grid_x, int(nx[0]) - grid_x)
    grid_y = np.minimum(grid_y, int(nx[1]) - grid_y)
    km = np.round(np.sqrt(grid_x**2.0 + grid_y**2.0)).astype(np.int)
    km = km.reshape((nx[0]*nx[1]))
    km = tf.convert_to_tensor(km, dtype=tf.int32)

    #P = tf.pow(tf.abs(rfft2d_tf(vel[...,0])), 2.0)
    P = rfft2d_tf(vel[...,0])
    P = tf.reshape(P, [nx[0]*nx[1]])
    Eu = tf.unsorted_segment_sum(P, km, kmax+1)
  
    P = rfft2d_tf(vel[...,1])
    #P = tf.pow(tf.abs(rfft2d_tf(vel[...,1])), 2.0)
    P = tf.reshape(P, [nx[0]*nx[1]])
    Ev = tf.unsorted_segment_sum(P, km, kmax+1)

    return Eu, Ev

  elif dim == 3:
    kmax = int(round(np.sqrt((1-nxc[0])**2 + (1-nxc[1])**2 + (1-nxc[2])**2)) + 1)
    Eu = np.zeros(kmax)
    Ev = np.zeros(kmax)
    Ew = np.zeros(kmax)
  
    grid_x = np.linspace(0, nx[0]-1, nx[0])
    grid_y = np.linspace(0, nx[1]-1, nx[1])
    grid_z = np.linspace(0, nx[2]-1, nx[2])
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    grid_x = np.minimum(grid_x, int(nx[0]) - grid_x) 
    grid_y = np.minimum(grid_y, int(nx[1]) - grid_y)
    grid_z = np.minimum(grid_z, int(nx[2]) - grid_z)
    km = np.round(np.sqrt(grid_x**2.0 + grid_y**2.0 + grid_z**2.0)).astype(np.int)
    km = km.reshape((nx[0]*nx[1]*nx[2]))
    km = tf.convert_to_tensor(km, dtype=tf.int32)
  
    P = tf.pow(tf.abs(rfft3d_tf(vel[...,0])), 2.0)
    P = tf.reshape(P, [nx[0]*nx[1]*nx[2]])
    Eu = tf.unsorted_segment_sum(P, km, kmax+1)

    P = tf.pow(tf.abs(rfft3d_tf(vel[...,1])), 2.0)
    P = tf.reshape(P, [nx[0]*nx[1]*nx[2]])
    Ev = tf.unsorted_segment_sum(P, km, kmax+1)

    P = tf.pow(tf.abs(rfft3d_tf(vel[...,2])), 2.0)
    P = tf.reshape(P, [nx[0]*nx[1]*nx[2]])
    Ew = tf.unsorted_segment_sum(P, km, kmax+1)

    return Eu, Ev, Ew
  
def rfft2d_tf(dat):
  dat = tf.complex(dat, tf.zeros_like(dat))
  fft = tf.spectral.fft2d(dat)
  fft = tf.pow(tf.abs(fft), 2.0)
  return fft
 
def rfft3d_tf(dat):
  dat = tf.complex(dat, tf.zeros_like(dat))
  fft = tf.spectral.fft3d(dat)
  fft = tf.sqrt(tf.square(tf.real(fft)) + tf.square(tf.imag(fft)))
  return fft

def gradient_np(dat, dx): 
  """ Compute the gradient spectrum of a tensor (same as gradient in matlab):
  Args:
    dat: NP Tensor of shape [H, W] or [H, W, D]
    dx:  list of scalar values indicating spacing
  Returns:
    grad: list Tensors of shape [H-2, W-2] or [H, W, D]
    #############################################
    # This implementation cuts off the edges of #
    # the input data. This is diffrent then     # 
    # matlab but needs to be this way to keep   #
    # consistant with tensorflow implementation #
    #############################################
  """
  if len(dx) == 2:
    grad_x, grad_y = np.gradient(dat, *dx)
    grad_x = grad_x[1:-1,1:-1]
    grad_y = grad_y[1:-1,1:-1]
    return grad_x, grad_y
  if len(dx) == 3:
    grad_x, grad_y, grad_z = np.gradient(dat, *dx)
    grad_x = grad_x[1:-1,1:-1,1:-1]
    grad_y = grad_y[1:-1,1:-1,1:-1]
    grad_z = grad_z[1:-1,1:-1,1:-1]
    return grad_x, grad_y, grad_z

def gradient_tf(dat, dx): 
  """ Compute the gradient spectrum of a tensor (same as gradient in matlab):
  Args:
    dat: TF Tensor of shape [H, W] or [H, W, D]
    dx:  list of scalar values indicating spacing
  Returns:
    grad: list of Tensors of shape [H-2, W-2] or [H, W, D]
    #############################################
    # This implementation cuts off the edges of #
    # the input data. This is different then    # 
    # matlab and may be fixed later if needed   #
    #############################################
  """
  # add fake batch
  dat = tf.expand_dims(dat, axis=0)

  # add fake pixel length
  dat = tf.expand_dims(dat, axis=-1)

  if len(dx) == 2:
    # make weight for x difference
    weight_x_np = np.zeros([3,1,1,1], dtype=np.float32)
    weight_x_np[0,0,0,0] = (1.0 / dx[0]) * (-1.0/2.0)
    weight_x_np[1,0,0,0] = 0.0 
    weight_x_np[2,0,0,0] = (1.0 / dx[0]) * (1.0/2.0)
    weight_x = tf.convert_to_tensor(weight_x_np)

    # make weight for y divergence
    weight_y_np = np.zeros([1,3,1,1], dtype=np.float32)
    weight_y_np[0,0,0,0] = (1.0 / dx[1]) * (-1.0/2.0)
    weight_y_np[0,1,0,0] = 0.0 
    weight_y_np[0,2,0,0] = (1.0 / dx[1]) * (1.0/2.0)
    weight_y = tf.convert_to_tensor(weight_y_np)

    # calc gradientes
    grad_x = _simple_conv_2d(dat, weight_x)
    grad_y = _simple_conv_2d(dat, weight_y)
    grad_x = grad_x[0,:,1:-1,0] # cut of edges
    grad_y = grad_y[0,1:-1,:,0]

    return grad_x, grad_y

  if len(dx) == 3:
    # make weight for x difference
    weight_x_np = np.zeros([3,1,1,1,1], dtype=np.float32)
    weight_x_np[0,0,0,0,0] = (1.0 / dx[0]) * (-1.0/2.0)
    weight_x_np[1,0,0,0,0] = 0.0 
    weight_x_np[2,0,0,0,0] = (1.0 / dx[0]) * (1.0/2.0)
    weight_x = tf.convert_to_tensor(weight_x_np)

    # make weight for y divergence
    weight_y_np = np.zeros([1,3,1,1,1], dtype=np.float32)
    weight_y_np[0,0,0,0,0] = (1.0 / dx[1]) * (-1.0/2.0)
    weight_y_np[0,1,0,0,0] = 0.0 
    weight_y_np[0,2,0,0,0] = (1.0 / dx[1]) * (1.0/2.0)
    weight_y = tf.convert_to_tensor(weight_y_np)

    # make weight for y divergence
    weight_z_np = np.zeros([1,1,3,1,1], dtype=np.float32)
    weight_z_np[0,0,0,0,0] = (1.0 / dx[2]) * (-1.0/2.0)
    weight_z_np[0,0,1,0,0] = 0.0 
    weight_z_np[0,0,2,0,0] = (1.0 / dx[2]) * (1.0/2.0)
    weight_z = tf.convert_to_tensor(weight_z_np)

    # calc gradientes
    grad_x = _simple_conv_3d(dat, weight_x)
    grad_y = _simple_conv_3d(dat, weight_y)
    grad_z = _simple_conv_3d(dat, weight_z)
    grad_x = grad_x[0,:,1:-1,1:-1,0]
    grad_y = grad_y[0,1:-1,:,1:-1,0]
    grad_z = grad_z[0,1:-1,1:-1,:,0]

    return grad_x, grad_y, grad_z

def aij_np(field, dx):
  if len(dx) == 2:
    a12, a11 = gradient_np(field[:,:,0], dx)
    a22, a21 = gradient_np(field[:,:,1], dx)

    aij = np.zeros([2,2] + list(a12.shape))

    aij[0,0] = a11
    aij[0,1] = a12
    aij[1,0] = a21
    aij[1,1] = a22

    return aij

  if len(dx) == 3:
    a11, a12, a13 = gradient_np(field[:,:,:,0], dx)
    a21, a22, a23 = gradient_np(field[:,:,:,1], dx)
    a31, a32, a33 = gradient_np(field[:,:,:,2], dx)
    #a12, a11, a13 = gradient_np(field[:,:,:,0], dx)
    #a22, a21, a23 = gradient_np(field[:,:,:,1], dx)
    #a32, a31, a33 = gradient_np(field[:,:,:,2], dx)

    aij = np.zeros([3,3] + list(a12.shape))

    aij[0,0] = a11
    aij[0,1] = a12
    aij[0,2] = a13
    aij[1,0] = a21
    aij[1,1] = a22
    aij[1,2] = a23
    aij[2,0] = a31
    aij[2,1] = a32
    aij[2,2] = a33

    return aij

def aij_tf(field, dx):
  if len(dx) == 2:
    a12, a11 = gradient_tf(field[:,:,0], dx)
    a22, a21 = gradient_tf(field[:,:,1], dx)
    a12 = tf.reshape(a12, [1,1] + int_shape(a12))
    a11 = tf.reshape(a11, [1,1] + int_shape(a11))
    a22 = tf.reshape(a22, [1,1] + int_shape(a22))
    a21 = tf.reshape(a21, [1,1] + int_shape(a21))

    aij_temp_a = tf.concat([a11, a12], axis=1)
    aij_temp_b = tf.concat([a21, a22], axis=1)
    aij = tf.concat([aij_temp_a, aij_temp_b], axis=0)
    return aij

  if len(dx) == 3:
    a12, a11, a13 = gradient_tf(field[:,:,:,0], dx)
    a22, a21, a23 = gradient_tf(field[:,:,:,1], dx)
    a32, a31, a33 = gradient_tf(field[:,:,:,2], dx)

    aij_tmp = []
    aij_tmp.append(tf.stack([a11, a12, a13], axis=0))
    aij_tmp.append(tf.stack([a21, a22, a23], axis=0))
    aij_tmp.append(tf.stack([a31, a32, a33], axis=0))
    aij = tf.stack(aij_tmp, axis=0)

    return aij

def aij_to_intermittency(aij, bins):
    dudx_np = aij[0,0]
    dudx_np_sq = np.power(dudx_np, 2.0)
    norm_np = np.sqrt(np.sum(dudx_np_sq)/np.prod(dudx_np_sq.shape))
    zdns_np = dudx_np / norm_np
    zdns_np = zdns_np.reshape((np.prod(zdns_np.shape)))
    [cnt_np, ebins_np] = np.histogram(zdns_np, bins)
    ebins_np = ebins_np[1:] # cutting off edge to line up bins and bin edges (puts things slightly off)
    zdns_np_plot = cnt_np/np.trapz(cnt_np, ebins_np)
    return ebins_np, zdns_np_plot

def structure_functions_np(vel, dx, distances, orders):
  u = vel[...,0] 
  deltau = np.zeros((len(orders), len(distances)))
  for n in range(len(orders)):
    for dist in range(len(distances)):
      veldiff = u[distances[dist]:] - u[:-distances[dist]]
      veldiffsq = np.power(np.abs(veldiff), orders[n])
      deltau[n, dist] = np.sum(veldiffsq)/(np.prod(veldiffsq.shape))
      #deltau[n, dist] = np.sum(veldiffsq)

  dn = deltau
  zeta = np.zeros(len(orders))
  x = distances * dx[0]
 
  for i in range(len(orders)): 
    y = dn[i,:]
    p = np.polyfit(np.log(x), np.log(y), 1) 
    m = p[0]
    b = np.exp(p[1])
    plt.scatter(x, y)
    plt.plot(x, b*np.power(x, m))
    plt.show()
    zeta[i]=m

  return zeta

def structure_functions_tf(vel, dx, distances, orders):
  u = vel[...,0] 
  dn = []
  for n in range(len(orders)):
    dn_tmp = []
    for dist in range(len(distances)):
      delta_u = u[distances[dist]:] - u[:-distances[dist]]
      delta_u_sq = tf.pow(tf.abs(delta_u), orders[n] + 1)
      dn_tmp.append(tf.reduce_sum(delta_u_sq)/np.prod(np.array(int_shape(delta_u_sq))))
    dn_tmp = tf.stack(dn_tmp, axis=0)
    dn.append(dn_tmp)
  dn = tf.stack(dn, axis=-1)
  return dn

def q_r_np(vel, dx, coarse_grains=[0,4,16]): # coarse grain with 1 is not doing coarse grain
  if len(dx) == 2: 
    vel_3d = np.zeros(list(vel.shape[:2]) + [3] + [3])
    vel_3d[:,:,0,:] = vel
    vel_3d[:,:,1,:] = vel
    vel_3d[:,:,2,:] = vel
    dx = [0.1,0.1,0.1]
    full_aij = aij_np(vel_3d, dx)
  else: 
    full_aij = aij_np(vel, dx)


  aij = {}
  for n in coarse_grains:
    if n == 0:
      aij[n] = full_aij
    else:
      aij[n] = np.zeros(list(full_aij.shape[:2]) + [x - n for x in full_aij.shape[2:]])
      # TODO fix this slow part!
      if len(dx) == 2:
        for i in range(aij[n].shape[2]):
          for j in range(aij[n].shape[3]):
            aij[n][:,:,i,j] = np.mean(full_aij[:,:,
                               i:i+n,
                               j:j+n],
                               axis=(2,3))
      if len(dx) == 3:
        for i in range(aij[n].shape[2]):
          for j in range(aij[n].shape[3]):
            for k in range(aij[n].shape[4]):
              aij[n][:,:,i,j,k] = np.mean(full_aij[:,:,
                                   i:i+n,
                                   j:j+n,
                                   k:k+n], 
                                   axis=(2,3,4))
              aij[n][:,:,i,j,k] = np.mean(full_aij[:,:,
                                   i:i+n,
                                   j:j+n,
                                   k:k+n], 
                                   axis=(2,3,4))


  rij = {}
  rm   = {}
  qw  = {}
  qm  = {}
  for n in coarse_grains:
    rij[n] = np.zeros_like(aij[n])
    rm[n]   = np.zeros(aij[n].shape[2:])
    qw[n]  = np.zeros(aij[n].shape[2:])
    qm[n]  = np.zeros(aij[n].shape[2:])
    for i in range(len(dx)):
      for j in range(len(dx)):
        rij[n][i,j] = 0.5 * (aij[n][i,j] - aij[n][j,i])
        qm[n] = qm[n] - (0.5 * aij[n][i,j] * aij[n][j,i])
        qw[n] = qw[n] - (0.5 * rij[n][i,j] * rij[n][j,i])
        for k in range(len(dx)):
          rm[n] = rm[n] - ((1./3.) * aij[n][i,j]
                                   * aij[n][j,k]
                                   * aij[n][k,i])

  qw_mean = {}
  r = {}
  q = {}
  for n in coarse_grains:
    qw_mean[n] = np.mean(qw[n])
    r[n] = rm[n] / np.power(qw_mean[n], 1.5)
    q[n] = qm[n] / qw_mean[n]

  return q, r

def plot_structure_function(dn, dx, distances, orders, save_path=None):
  zeta = np.zeros(len(orders))
  x = dx[0] * np.array(distances)
  for i in range(len(orders)):
    y = dn[i,:]
    p = np.polyfit(np.log(x), np.log(y), 1)
    m = p[0]
    b = np.exp(p[1])
    zeta[i] = m

  plt.scatter(orders[1:], zeta[1:])
  plt.plot(orders, orders/3)
  plt.show()


def test_energy_spectrum(dim=2):
  if dim == 2:
    # random function
    stream_hr = h5py.File('00000.h5')
    key_hr = stream_hr.keys()[4]
    value = np.array(stream_hr[key_hr])[0,:,:,0:2]
    
    # numpy energy spectrum
    Eu_s, Ev_s = energy_spectrum_np(value)
    E_s = Eu_s + Ev_s
    plt.loglog(np.arange(E_s.shape[0])+1, E_s, label='numpy')

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [1024, 1024 ,2], name="input_flow")
    Eu, Ev = energy_spectrum_tf(flow)
    Eu_np, Ev_np = sess.run([Eu, Ev], feed_dict={flow: value})
    E_np = Eu_np + Ev_np
    plt.loglog(np.arange(E_np.shape[0])+1, E_np, label='tensorflow')

    # plot 
    plt.title("2D energy spectrum of numpy and tensorflow")
    plt.legend()
    plt.show()

  if dim == 3:
    grid_x = np.linspace(0, 15, 16)
    grid_y = np.linspace(0, 15, 16)
    grid_z = np.linspace(0, 15, 16)
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    value = np.sin(grid_x + grid_y + grid_z) + np.cos(grid_x) + np.sin(2*grid_z)
    value = np.stack([value, value, value], axis=-1)
 
    # numpy energy spectrum
    Eu_s, Ev_s, Ew_s = energy_spectrum_np(value)
    E_s = Eu_s + Ev_s + Ew_s
    plt.plot(E_s, label='numpy')

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [16,16,16,3], name="input_flow")
    Eu, Ev, Ew = energy_spectrum_tf(flow)
    
    Eu_np, Ev_np, Ew_np = sess.run([Eu, Ev, Ew], feed_dict={flow: value})
    E_np = Eu_np + Ev_np + Ew_np
    plt.plot(E_np, label='tensorflow')
   
    # plot 
    plt.title("3D energy spectrum of numpy and tensorflow")
    plt.legend()
    plt.show()

def test_intermittency_plot(dim=2):
  bins = 100
  if dim == 2:
    dx = [0.1, 0.1]
    # random function
    stream_hr = h5py.File('00000.h5')
    key_hr = stream_hr.keys()[4]
    value = np.array(stream_hr[key_hr])[0,:,:,0:2]
    
    # numpy energy spectrum
    numpy_aij = aij_np(value, dx)
    ebins_np, zdns_np_plot = aij_to_intermittency(numpy_aij, bins)
    plt.semilogy(ebins_np, zdns_np_plot, label='numpy')

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [1024,1024,2], name="input_flow")
    tensorflow_aij = aij_tf(flow, dx)
    tensorflow_aij_tf = sess.run(tensorflow_aij, feed_dict={flow: value})
    ebins_tf, zdns_tf_plot = aij_to_intermittency(tensorflow_aij_tf, bins)
    plt.semilogy(ebins_tf, zdns_tf_plot, label='tensorflow')

    # plot 
    plt.title("2D intermittency of numpy and tensorflow")
    plt.legend()
    plt.show()


  if dim == 3:
    dx = [0.1, 0.1, 0.1]
    grid_x = np.linspace(0, 15, 16)
    grid_y = np.linspace(0, 15, 16)
    grid_z = np.linspace(0, 15, 16)
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    value = np.sin(grid_x + grid_y + grid_z) + np.cos(grid_x) + np.sin(2*grid_z)
    value = np.stack([value, value, value], axis=-1)
 
    # numpy energy spectrum
    numpy_aij = aij_np(value, dx)
    ebins_np, zdns_np_plot = aij_to_intermittency(numpy_aij, bins)
    plt.semilogy(ebins_np, zdns_np_plot, label='numpy')

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [16,16,16,3], name="input_flow")
    tensorflow_aij = aij_tf(flow, dx)
    tensorflow_aij_tf = sess.run(tensorflow_aij, feed_dict={flow: value})
    ebins_tf, zdns_tf_plot = aij_to_intermittency(tensorflow_aij_tf, bins)
    plt.semilogy(ebins_tf, zdns_tf_plot, label='tensorflow')
    
    # plot 
    plt.title("3D intermittency of numpy and tensorflow")
    plt.legend()
    plt.show()

def test_gradient_plot(dim=2):
  if dim == 2:
    dx = [0.1, 0.1]
    # random function
    stream_hr = h5py.File('00000.h5')
    key_hr = stream_hr.keys()[4]
    value = np.array(stream_hr[key_hr])[0,:,:,0]
    
    # numpy gradient
    numpy_grad_x, numpy_grad_y = gradient_np(value, dx)

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [1024,1024], name="input_flow")
    tensorflow_grad_x, tensorflow_grad_y = gradient_tf(flow, dx)
    tensorflow_grad_x_np = sess.run(tensorflow_grad_x, feed_dict={flow: value})

    # plot
    plt.title("2D diff in gradient of numpy and tensorflow")
    plt.imshow(np.abs(tensorflow_grad_x_np - numpy_grad_x))
    plt.show()

  if dim == 3:
    # random function
    dx = [0.1, 0.1, 0.1]
    grid_x = np.linspace(0, 15, 16)
    grid_y = np.linspace(0, 15, 16)
    grid_z = np.linspace(0, 15, 16)
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    value = np.sin(grid_x + grid_y + grid_z) + np.cos(grid_x) + np.sin(2*grid_z)
 
    # numpy gradient
    numpy_grad_x, numpy_grad_y, numpy_grad_z = gradient_np(value, dx)

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [16, 16, 16], name="input_flow")
    tensorflow_grad_x, tensorflow_grad_y, tensorflow_grad_z = gradient_tf(flow, dx)
    tensorflow_grad_x_np = sess.run(tensorflow_grad_x, feed_dict={flow: value})

    # plot
    plt.title("3D diff in gradient of numpy and tensorflow")
    plt.imshow(np.abs(tensorflow_grad_x_np[0] - numpy_grad_x[0]))
    plt.show()

def test_fft(dim=2):
  if dim == 2:
    dx = [0.1, 0.1]
    # random function
    stream_hr = h5py.File('00000.h5')
    key_hr = stream_hr.keys()[4]
    value = np.array(stream_hr[key_hr])[0,:,:,0]
    
    # numpy gradient
    spectrum_np = np.fft.fftn(value)
    spectrum_np = np.power(np.abs(spectrum_np), 2.0)

    # tensorflow energy spectrum
    sess = tf.InteractiveSession()
    flow = tf.placeholder(tf.float32, [1024,1024], name="input_flow")
    spectrum_tf = rfft2d_tf(flow)
    spectrum_tf_np = sess.run(spectrum_tf, feed_dict={flow: value})

    # plot
    plt.imshow(spectrum_np)
    plt.show()
    plt.imshow(spectrum_tf_np)
    plt.show()
    plt.imshow(np.abs(spectrum_np - spectrum_tf_np))
    plt.show()

def test_structure_functions(dim=3):
  if dim == 3:
    dx = [0.05, 0.05, 0.05]
    orders = np.arange(10) + 1
    distances = (np.arange(10)) + 2

    # random function
    vel = np.load('0000.npy')[0,:,:,:,0:3]
 
    # numpy gradient
    zeta = structure_functions_np(vel, dx, distances, orders)

    struct_sim = np.zeros(len(orders))
    mu = 0.25 
    for n in range(len(orders)):
      struct_sim[n]= 1./3.*orders[n]*(1.0-(1.0/6.0)*mu*(orders[n]-3.0));
    plt.scatter(orders[1:], zeta[1:])
    plt.plot(orders, struct_sim)
    plt.plot(orders, orders/3)
    plt.show()

def plot_contour(axi, x, y, label, color):
  v = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100]
  edges = np.arange(-10, 10, .5)
  bounds = 10
  N, xi_edges, yi_edges = np.histogram2d(y, x, bins=(edges, edges), normed=True)
  #N = np.flip(N, axis=0)
  #N = np.flip(N, axis=1)
  xi_width = xi_edges[2] - xi_edges[1]
  yi_width = yi_edges[2] - yi_edges[1]
  xc = xi_edges + xi_width/2
  xc = xc[:-1]
  yc = yi_edges + yi_width/2
  yc = yc[:-1]

  xr = np.linspace(-10, 10, 1000)
  xq = -np.power(27.0/4.0 * np.power(xr, 2.0), (1.0/3.0))

  axi.plot(xr, xq)
  #axi.scatter(x, y)
  cnt = axi.contour(xc, yc, N, levels=v, linewidths=1.0, colors=color)
  h1, _ = cnt.legend_elements()
  return h1

  #plt.xlim((-10, 10))
  #plt.ylim((-10, 10))
  #xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
  #xi, yi = np.mgrid[-bounds:bounds:nbins*1j, -bounds:bounds:nbins*1j]
  #zi = k(np.vstack([xi.flatten(), yi.flatten()]))
  #zi = np.log(k(np.vstack([xi.flatten(), yi.flatten()])))
  #plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
  #plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

def test_q_r_plot(dim=2):
  if dim == 2:
    dx = [0.1, 0.1]

    # read flow file
    stream_hr = h5py.File('00000.h5')
    key_hr = stream_hr.keys()[4]
    value = np.array(stream_hr[key_hr])[0,:128,:128,1:3]
 
    # numpy gradient
    q, r = q_r_np(value, dx, coarse_grains=[0])
 
    # plot
    plt.scatter(r[0].flatten(), q[0].flatten(), s=0.1)
    xr = np.linspace(-10, 10, 1000)
    xq = -np.power(27.0/4.0 * np.power(xr, 2.0), (1.0/3.0))
    plt.plot(xr, xq)
    plt.show()


  elif dim == 3:
    dx = [0.1, 0.1, 0.1]

    # read flow file
    vel = np.load('0000.npy')[0,:,:,:,0:3]
 
    # numpy gradient
    q, r = q_r_np(vel, dx, coarse_grains=[0,2,4])
 
    # plot
    plt.scatter(r[0].flatten(), q[0].flatten(), s=0.1)
    xr = np.linspace(-10, 10, 1000)
    xq = -np.power(27.0/4.0 * np.power(xr, 2.0), (1.0/3.0))
    plt.plot(xr, xq)
    plt.show()


def diagnostics_np(net_vel, true_vel, save_dir='./diagnosticsOliver', iteration=0, pos=[0,0,0], dx=[0.1, 0.1, 0.1], diagnostics=['spectrum', 'intermittency', 'structure_functions', 'QR', 'image']):

  save_path = (save_dir + '/iter_' + str(iteration).zfill(4) + '_pos_'
               + str(pos[0]) + '_'
               + str(pos[1]) + '_'
               + str(pos[2]) + '_')

  if 'spectrum' in diagnostics:
    net_e = np.sum(energy_spectrum_np(net_vel), axis=0)
    true_e = np.sum(energy_spectrum_np(true_vel), axis=0)
    net_e = net_e/true_e[1]
    true_e = true_e/true_e[1]
    np.savez(save_path + 'energy_spectrum', net_e=net_e, true_e=true_e)
    x = np.arange(1, net_e.shape[0])
    plt.loglog(x, net_e[1:],      label='Lat-Net flow')
    plt.loglog(x, true_e[1:],     label='true flow')
    plt.loglog(x, np.power(x, -(5.0/3.0)), label='-5/3 power rule')
    plt.title("Energy Spectrum")
    plt.ylabel("Energy")
    plt.legend(loc=0)
    plt.savefig(save_path + 'energy_spectrum.png')
    #plt.show()
    plt.close()

  if 'intermittency' in diagnostics:
    bins = np.arange(-6.5, 6.5, 13.0/100)
    net_aij = aij_np(net_vel, dx)
    true_aij = aij_np(true_vel, dx)
    net_ebins, net_zdns_plot = aij_to_intermittency(net_aij, bins)
    true_ebins, true_zdns_plot = aij_to_intermittency(true_aij, bins)
    np.savez(save_path + 'intermittency', net_ebins=net_ebins, net_zdns_plot=net_zdns_plot, true_ebins=true_ebins, true_zdns_plot=true_zdns_plot)
    plt.semilogy(net_ebins, net_zdns_plot, label='network flow')
    plt.semilogy(true_ebins, true_zdns_plot, label='true flow')
    plt.legend(loc=0)
    plt.title("Intermittency")
    plt.savefig(save_path + 'intermittency.png')
    plt.close()

  if 'structure_functions' in diagnostics:
    """
    orders = np.arange(10)
    distances = np.arange(2,30,2)
    dn = structure_functions_np(net_vel, dx, distances, orders)
    plot_structure_function(dn, dx, distances, orders, save_path + 'structure_functions_network.png')
    plt.close()
    dn = structure_functions_np(true_vel, dx, distances, orders)
    plot_structure_function(dn, dx, distances, orders, save_path + 'structure_functions_true.png')
    plt.close()
    """

  if 'QR' in diagnostics:
    net_q, net_r = q_r_np(net_vel, dx, coarse_grains=[0, 8, 32])
    true_q, true_r = q_r_np(true_vel, dx, coarse_grains=[0, 8, 32])
    np.savez(save_path + 'qr_data', net_q_0 =net_q[0],  net_q_8=net_q[8],   net_q_32=net_q[32], 
                                    net_r_0 =net_r[0],  net_r_8=net_r[8],   net_r_32=net_r[32], 
                                    true_q_0=true_q[0], true_q_8=true_q[8], true_q_32=true_q[32],
                                    true_r_0=true_r[0], true_r_8=true_r[8], true_r_32=true_r[32])
    """
    plot_contour(net_r[0].flatten(), net_q[0].flatten(), label="network flow", color='green')
    plot_contour(true_r[0].flatten(), true_q[0].flatten(), label="true flow", color='red')
    plt.title("Intermittency")
    plt.xlabel("R/Qw")
    plt.ylabel("Q/Qw")
    plt.legend(loc=0)
    #plt.show()
    plt.savefig(save_path + 'qr.png')
    plt.close()
    """
 
  if 'image' in diagnostics:
    net_norm = vel_to_norm(net_vel)
    true_norm = vel_to_norm(true_vel)
    net_norm = net_norm[0]
    true_norm = true_norm[0]
    np.savez(save_path + 'image', net_norm=net_norm, true_norm=true_norm)

"""
test_energy_spectrum(dim=2) 
test_energy_spectrum(dim=3) 
test_intermittency_plot(dim=2)
test_intermittency_plot(dim=3)
test_gradient_plot(dim=2)
test_gradient_plot(dim=3)
test_structure_functions(dim=3)
test_q_r_plot(dim=2)
test_q_r_plot(dim=3)
"""


