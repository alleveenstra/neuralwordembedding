package com.github.alleveenstra.neuralwordembedding.tools;

import org.junit.Test;

import static com.github.alleveenstra.neuralwordembedding.tools.util.Validation.isValidSentence;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class ValidationTest {
    @Test
    public void isValid() {
        assertThat(isValidSentence("Dit, is: een test!"), equalTo(true));
        assertThat(isValidSentence("Dit is één test."), equalTo(true));
        assertThat(isValidSentence("Dit is één rʌʃə test."), equalTo(false));
    }
}
