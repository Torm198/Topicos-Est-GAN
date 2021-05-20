import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, LeakyReLU, Dropout, Conv2DTranspose
import matplotlib.pyplot as plt


def imp_imagem(diretorio):
    arquivos = os.listdir(diretorio)
    dados = []
    for imagem in arquivos:
        imp = img_to_array(load_img(diretorio + imagem, color_mode='grayscale'))
        if np.shape(imp) == (103, 96, 1):
            dados.append(imp[7:, :, :] / 255)
    dados = np.asarray(dados)
    return dados


Banco_real = imp_imagem('Fotos/Teste/')

######################################

discriminador = Sequential()
discriminador.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(96, 96, 1)))
discriminador.add(LeakyReLU(alpha=0.2))
discriminador.add(Dropout(0.4))
discriminador.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
discriminador.add(LeakyReLU(alpha=0.2))
discriminador.add(Dropout(0.4))
discriminador.add(Flatten())
discriminador.add(Dense(1, activation='sigmoid'))
opt = Adam(lr=0.0002, beta_1=0.5)
discriminador.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#######################################
gerador = Sequential()
# gerando uma imagem 6x6
gerador.add(Dense(128 * 6 * 6, input_dim=100))
gerador.add(LeakyReLU(alpha=0.2))
gerador.add(Reshape((6, 6, 128)))
# upsample para 24x24
gerador.add(Conv2DTranspose(128, (4, 4), strides=(4, 4), padding='same'))
gerador.add(LeakyReLU(alpha=0.2))

# upsample para 96x96
gerador.add(Conv2DTranspose(128, (4, 4), strides=(4, 4), padding='same'))
gerador.add(LeakyReLU(alpha=0.2))

# layer de saida
gerador.add(Conv2D(1, (6, 6), activation="sigmoid", padding="same"))


######################################

def cria_gan(modelo_g=gerador, modelo_d=discriminador):
    global opt
    modelo_d.trainable = False
    GAN = Sequential()
    GAN.add(modelo_g)
    GAN.add(modelo_d)
    GAN.compile(loss='binary_crossentropy', optimizer=opt)
    return GAN


GAN = cria_gan()


#################################
def imagens_falsas(gen, n):
    x = np.random.randn(n * 100)
    x = np.reshape(x, (n, 100))
    saida = gen.predict(x)
    label = np.zeros((n, 1))
    return saida, label


def amostra_imagens_reais(dados, n):
    ix = np.random.randint(0, dados.shape[0], n)
    X = dados[ix]
    y = np.ones((n, 1))
    return X, y


def salvar_imagem(gen, epoch):
    ex = imagens_falsas(gen, 1)
    plt.title('Epoch ' + str(epoch + 1))
    plt.axis('off')

    plt.imshow(ex[0][0, :, :, 0], cmap='gray')
    plt.savefig(f'Fotos/Digitais Geradas/Epoch{epoch + 1}.png')
    plt.close()


##########################
n_batch = 100
n_epochs = 1000

dloss_list = [0.0] * n_epochs
dacc_list = [0.0] * n_epochs
gloss_list = [0.0] * n_epochs
bat_por_epo = int(2000 / n_batch)
meia_amostra = int(n_batch / 2)

for i in range(n_epochs):
    for j in range(bat_por_epo):
        x_real, y_real = amostra_imagens_reais(Banco_real, meia_amostra)
        x_falso, y_falso = imagens_falsas(gerador, meia_amostra)
        X, y = np.vstack((x_real, x_falso)), np.vstack((y_real, y_falso))
        d_loss, d_acc = discriminador.train_on_batch(X, y)
        semente = np.random.randn(n_batch * 100).reshape(n_batch, 100)
        y_semente = np.ones((n_batch, 1))
        g_loss = GAN.train_on_batch(semente, y_semente)

    gloss_list[i] = g_loss
    dloss_list[i] = d_loss
    dacc_list[i] = d_acc
    if (i + 1) % 10 == 0:
        salvar_imagem(gerador, i)

label_x = range(1, n_epochs+1)
plt.plot(label_x, gloss_list)
plt.plot(label_x, dloss_list)
plt.legend(['Loss Gerador', 'Loss Discriminador'])
plt.savefig('modelo1 loss.png')
plt.close()

plt.plot(label_x, dacc_list)
plt.savefig('modelo1 acc.png')
plt.close()