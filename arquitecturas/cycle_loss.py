def define_composite_model_with_total_loss_and_metrics(g_model_1, d_model, g_model_2, image_shape, L=0.05, M=0.15, N=0.8):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    output_f = g_model_2(gen1_out)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    
    model.compile(loss=lambda y_true, y_pred: total_loss(y_true, y_pred, L, M, N), 
                  optimizer=opt,
                  metrics=['mse', 'mae'])  
    
    return model
