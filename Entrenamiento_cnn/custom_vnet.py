from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    Dropout,
    SpatialDropout3D,
    UpSampling3D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
    ZeroPadding3D,
    Lambda,
    Cropping3D
)

import imag3D.CNN_superficie.cnnsurf_funcs as fu


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv3DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling3D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv3D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv3D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    inp_1_conv, inp_2_conv = fu.match_tensor_shapes(inp_1_conv, inp_2_conv)
    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv3D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    inp_1, h = fu.match_tensor_shapes(inp_1, h)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    conv_below, attention_across = fu.match_tensor_shapes(conv_below, attention_across)
    return concatenate([conv_below, attention_across])


def conv3d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
    kernel_regularizer=None,
    bias_regularizer=None,
):

    if dropout_type == "spatial":
        DO = SpatialDropout3D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv3D(
        filters,
        kernel_size,
        strides=strides,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv3D(
        filters,
        kernel_size,
        strides=strides,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def custom_vnet(
    input_shape,
    pool_size=(2, 2, 2),
    conv_kernel_size=(3, 3, 3),
    conv_strides=(1, 1, 1),
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
    kernel_regularizer=None,
    bias_regularizer=None
):  # 'sigmoid' or 'softmax'

    """
    Customizable VNet architecture based on the work of
    Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi in
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

    Arguments:
    input_shape: 4D Tensor of shape (x, y, z, num_channels)

    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation

    activation (str): A keras.activations.Activation to use. ReLu by default.

    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers

    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part

    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off

    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block

    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]

    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network

    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]

    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block

    num_layers (int): Number of total layers in the encoder not including the bottleneck layer

    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation

    Returns:
    model (keras.models.Model): The built V-Net

    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"


    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]:

    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # si la entrada es (11,11, n_samples) y maxpool es (2,2, algo), hay que hacer cosas para cuadrar los tamaños
    # a esto lo llamo "caso_molesto"
    caso_molesto = input_shape[0] % 2 and pool_size[0] > 1

    # Build model
    inputs = Input(input_shape)
    if caso_molesto:  # chequea el resto de la division por 2
        # si en las direcciones espaciales la dimension es impar, se hace padding para volverlas pares
        # Si no, hay problemas en las skip connections ( en el caso de hacer pmax pooling en las direcciones espaciales)
        x = ZeroPadding3D(((0, 1), (0, 1), (0, 0)))(inputs)
    else:
        x = inputs

    if use_batch_norm:
        x = BatchNormalization()(x)

    down_layers = []
    for l in range(num_layers):
        x = conv3d_block(
            inputs=x,
            kernel_size=fu.match_kernel_to_input(conv_kernel_size, x),
            strides=conv_strides,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        down_layers.append(x)
        if pool_size is not None:
            x = MaxPooling3D(pool_size, padding='same')(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv3d_block(
        inputs=x,
        kernel_size=fu.match_kernel_to_input(conv_kernel_size, x),
        strides=conv_strides,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, pool_size, strides=pool_size, padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            # si no matchean
            if x.shape[3] != conv.shape[3]:
                x, conv = fu.match_tensor_shapes(x, conv)
            x = concatenate([x, conv]) # skip layer

        x = conv3d_block(
            inputs=x,
            kernel_size=fu.match_kernel_to_input(conv_kernel_size, x),
            strides=conv_strides,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    outputs = Conv3D(num_classes, (1, 1, 1), activation=output_activation,
                     kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)

    # agrego esto para manejar el tema de (11, 11) vs (12, 12)
    if caso_molesto:  # chequea el resto de la division por 2
        outputs = Cropping3D(((0, 1), (0, 1), (0, 0)))(outputs)
    else:
        outputs = outputs

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
