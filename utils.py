from gluonts.mx.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
from mxnet.gluon import nn


class ExtendedMQCNNEstimator(MQCNNEstimator):
    def __init__(self, *args, alpha: float, N: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.N = N

        # Define additional Dense layer
        self.additional_dense = nn.Dense(units=3, flatten=False)

    def hybrid_forward(self, F, x, additional_input, *args, **kwargs):
        # Get the output from the original decoder
        decoder_output = super().decoder.hybrid_forward(F, x, *args, **kwargs)

        # Apply Dense layer to decoder output
        decoder_output_expanded = self.additional_dense(decoder_output)

        # Apply softmax to each time step
        decoder_output_softmax = F.softmax(decoder_output_expanded, axis=-1)

        # Apply normalization
        decoder_output_normalized = self.alpha * decoder_output_softmax + (1 - self.alpha) / self.N

        # Element-wise multiplication and summation with additional input
        combined_output = F.sum(decoder_output_normalized * additional_input, axis=-1)

        # Add the original decoder output to each time step
        final_output = combined_output + decoder_output

        return final_output

