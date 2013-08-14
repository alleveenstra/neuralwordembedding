package com.github.alleveenstra.neuralwordembedding.tools;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class JBlasTest {

    @Test
    public void simple() {
        final DoubleMatrix eye = DoubleMatrix.eye(4);
        final DoubleMatrix mul = eye.mul(8d);
        assertThat(mul.max(), equalTo(8d));
    }

    @Test
    public void reshape() {
        final DoubleMatrix test = DoubleMatrix.rand(4, 4);
        final DoubleMatrix reshape = test.reshape(1, 16);
    }
}
