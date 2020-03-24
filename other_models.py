"""
#we are doing the steps outside of the keras sequential model here I guess 
#maybe b/c it doesn't support meta data addition? that doesn't relly make sense tbh b/c we can still contain all element in the model and just add output
#outside which would take a cancanation of meta data and last output from keras model.
#Anyway with that out of the way we are implementing the same process as above but this time directly feeding it the text data.
x=(Embedding(len(vectorizer.get_feature_names()) + 1, 
                    EMBEDDINGS_LEN,  # Embedding size
                                        weights=[embeddings_index],
                                                            input_length=MAX_SEQ_LENGHT,
                                                                                trainable=False))(text_data)

                                                                                #creat an LSTM later with dropout feed it x. Recurrent dropout b/w 0 and 1 represent the fraction of unitss we drop during linear tranformation or recurrent state.
                                                                                #Normal dropout refers to how much we drop during linear transform of inputs.
                                                                                x2 = ((LSTM(300, dropout=0.2, recurrent_dropout=0.2)))(x)
                                                                                #We concatenate our meta data with our lstm output
                                                                                x4 = concatenate([x2, meta_data])

                                                                                #add fuly connected relu layer I assume for output feed
                                                                                x5 = Dense(150, activation='relu')(x4)

                                                                                #another dropout this is inneficient can do it in one step
                                                                                x6 = Dropout(0.25)(x5)

                                                                                #applly batch normilization
                                                                                x7 = BatchNormalization()(x6)

                                                                                #output if softmax as per usual
                                                                                out=(Dense(len(set(y)), activation="softmax"))(x7)

                                                                                #now we add a model i guess why???
                                                                                model = Model(inputs=[text_data, meta_data ], outputs=out)

                                                                                #compile does computational graph inilization i think. We'll define our loss function, optimizer and what metics we want to see in summary here as well.
                                                                                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                                                                                #print a summary of what our network looks like
                                                                                print(model.summary())

                                                                                #traub model
                                                                                model.fit([X_train_sequences, df_cat_train], y_train, 
                                                                                                  epochs=12, batch_size=128, verbose=1, 
                                                                                                                              validation_split=0.1)

                                                                                                                              #test model
                                                                                                                              scores = model.evaluate([X_test_sequences, df_cat_test],y_test, verbose=1)
                                                                                                                              print("Accuracy:", scores[1])  
                                                                                                                              list_result.append(("LSTM with Multi-Input", scores[1]))

                                                                                                                              """


"""
#we are doing the steps outside of the keras sequential model here I guess 
#maybe b/c it doesn't support meta data addition? that doesn't relly make sense tbh b/c we can still contain all element in the model and just add output
#outside which would take a cancanation of meta data and last output from keras model.
#Anyway with that out of the way we are implementing the same process as above but this time directly feeding it the text data.
x=(Embedding(len(vectorizer.get_feature_names()) + 1, 
                    EMBEDDINGS_LEN,  # Embedding size
                                        weights=[embeddings_index],
                                                            input_length=MAX_SEQ_LENGHT,
                                                                                trainable=False))(text_data)

                                                                                #creat an LSTM later with dropout feed it x. Recurrent dropout b/w 0 and 1 represent the fraction of unitss we drop during linear tranformation or recurrent state.
                                                                                #Normal dropout refers to how much we drop during linear transform of inputs.
                                                                                x2 = ((LSTM(300, dropout=0.2, recurrent_dropout=0.2)))(x)
                                                                                #We concatenate our meta data with our lstm output
                                                                                x4 = concatenate([x2, meta_data])

                                                                                #add fuly connected relu layer I assume for output feed
                                                                                x5 = Dense(150, activation='relu')(x4)

                                                                                #another dropout this is inneficient can do it in one step
                                                                                x6 = Dropout(0.25)(x5)

                                                                                #applly batch normilization
                                                                                x7 = BatchNormalization()(x6)

                                                                                #output if softmax as per usual
                                                                                out=(Dense(len(set(y)), activation="softmax"))(x7)

                                                                                #now we add a model i guess why???
                                                                                model = Model(inputs=[text_data, meta_data ], outputs=out)

                                                                                #compile does computational graph inilization i think. We'll define our loss function, optimizer and what metics we want to see in summary here as well.
                                                                                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                                                                                #print a summary of what our network looks like
                                                                                print(model.summary())

                                                                                #traub model
                                                                                model.fit([X_train_sequences, df_cat_train], y_train, 
                                                                                                  epochs=12, batch_size=128, verbose=1, 
                                                                                                                              validation_split=0.1)

                                                                                                                              #test model
                                                                                                                              scores = model.evaluate([X_test_sequences, df_cat_test],y_test, verbose=1)
                                                                                                                              print("Accuracy:", scores[1])  
                                                                                                                              list_result.append(("LSTM with Multi-Input", scores[1]))

                                                                                                                              """

