TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1344
UPDATE_FREQ=1
BART_PATH=bart.large/model.pt

MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train Infilling-categorized-bin \
		--save-dir ./Infilling_checkpoint/ \
			    --max-tokens $MAX_TOKENS \
				--restore-file $BART_PATH \
				    --task translation \
					    --source-lang source --target-lang target \
						    --truncate-source \
							    --layernorm-embedding \
								    --share-all-embeddings \
									    --share-decoder-input-output-embed \
										    --reset-optimizer --reset-dataloader --reset-meters \
											    --required-batch-size-multiple 1 \
													--arch bart_large \
													    --criterion label_smoothed_cross_entropy \
														    --label-smoothing 0.1 \
															    --dropout 0.1 --attention-dropout 0.1 \
																    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
																	--clip-norm 0.1 \
																		    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
																			    --update-freq $UPDATE_FREQ \
																				    --skip-invalid-size-inputs-valid-test \
																					--keep-interval-updates 2\
																					--max-epoch 200\
																					--save-interval 20\
																					    --find-unused-parameters;
