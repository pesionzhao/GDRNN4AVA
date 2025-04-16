## GDRNN

Gradient-Driven Recurrent Neural Networks for High-Resolution Prestack AVA Inversion

### Structure

Inspired by diffusion, we reformulate the gradient descent process in traditional optimization methods as a Recurrent Neural Networks (RNNs).

### DataSet Prepare

Format of dataset for data-driven: Variables in square brackets [] represent data shape

```text
{
    'vp_aug': vp_label_aug, -> [layers, ntraces of augmented augmented]
    'vs_aug': vs_label_aug, -> [layers, ntraces of augmented augmented]
    'rho_aug': rho_label_aug, -> [layers, ntraces of augmented augmented]
    'vp_back': vp_bakcward ->  [layers, ntraces of augmented augmented]
    'vs_back': vp_bakcward, -> [layers, ntraces of augmented augmented]
    'rho_back': vp_bakcward, -> [layers, ntraces of augmented augmented]
    'seis_s': seis_near, -> [ntraces of augmented augmented, layers, number of angles of seis_near]
    'seis_m': seis_mid, -> [ntraces of augmented augmented, layers, number of angles of seis_mid]
    'seis_l': seis_far, -> [ntraces of augmented augmented, layers, number of angles of seis_far]
    'wavemat': wavemat, -> [layers, layers]
    'theta': [[angles of seis_near], [angles of seis_near], [angles of seis_far]] -> angles for near, mid and far section
}
```

Format of dataset for physice-driven:

```text
{
    'vp_back': vp_bakcward, ->  [layers, ntraces of observation]
    'vs_back': vp_bakcward, ->  [layers, ntraces of observation]
    'rho_back': vp_bakcward, ->  [layers, ntraces of observation]
    'seis_s': seis_near, ->  [ntraces of observation, layers, number of angles of seis_near]
    'seis_m': seis_mid, -> [ntraces of observation, layers, number of angles of seis_mid]
    'seis_l': seis_far, -> [ntraces of observation, layers, number of angles of seis_far]
    'wavemat': wavemat, -> [layers, layers]
    'theta': [[angles of seis_near], [angles of seis_near], [angles of seis_far]] -> angles for near, mid and far section
}
```

Train GDRNN script:

```bash
python train_GDRNN.py --cfg config/m_config.yaml --device cuda:0 --epoch 60 --step 5
```

Test GDRNN script:

```bash
python predict_GDRNN.py --cfg config/m_config.yaml --device cuda:0 --weight weights_dir/weights102/GDbest_56.pth --name custom_data --step 7
```

Inversion result for Marmousi dataset as follows, The dashed line is the log-well position. 

![](./inversion%20result.png)

By The Way, we give the original joint data-driven and physics-driven AVA inversion method code as comparison

Train original method script:

```bash
python train_original.py --cfg config/m_data.yaml --device cuda:0 --epoch 120
```

Predict original method script:

```bash
python predict_original.py --cfg config/m_data.yaml --device cuda:0 --weight your/weight/path --name custom_data
```