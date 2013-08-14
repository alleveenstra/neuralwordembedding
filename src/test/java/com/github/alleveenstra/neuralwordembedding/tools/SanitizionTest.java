package com.github.alleveenstra.neuralwordembedding.tools;

import com.github.alleveenstra.neuralwordembedding.tools.util.Sanitization;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class SanitizionTest {
    @Test
    public void test() {
        assertThat(Sanitization.sanitizeWord("06-14168992"), equalTo("*DD-DDDDDDDD*"));
        assertThat(Sanitization.sanitizeLine("De \"\"vixen\"\" was een \"boot\"?"), equalTo("De vixen was een boot?"));
    }
}
