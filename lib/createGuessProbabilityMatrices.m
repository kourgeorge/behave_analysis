function [guessTRr, guessTRnr] = createGuessProbabilityMatrices(realTRr, realTRnr, noiseVal)

[m,n] = size(realTRr);
guessTRr = (1-noiseVal).*realTRr + noiseVal.* rand(m,n);
guessTRnr = (1-noiseVal).*realTRnr + noiseVal.* rand(m,n);

end

