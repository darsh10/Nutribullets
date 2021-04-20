TASK=/data/rsg/nlp/darsh/aggregator/crawl_websites/NUT
for SPLIT in train dev
do
		for LANG in source target
				    do
							    python -m examples.roberta.multiprocessing_bpe_encoder \
										    --encoder-json bart.large/encoder.json \
											    --vocab-bpe bart.large/vocab.bpe \
												    --inputs "$TASK/$SPLIT.$LANG" \
													    --outputs "$TASK/$SPLIT.bpe.$LANG" \
														    --workers 60 \
															    --keep-empty;
								  done
						  done

fairseq-preprocess \
		  --source-lang "source" \
		    --target-lang "target" \
			  --trainpref "${TASK}/train.bpe" \
			    --validpref "${TASK}/dev.bpe" \
				  --destdir "Infilling-categorized-bin/" \
				    --workers 60 \
					  --srcdict bart.large/dict.txt \
					    --tgtdict bart.large/dict.txt;
