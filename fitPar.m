% This script minimizes the difference between a desired T1, T2 and HU
% number combination and fit based predictions from phantom additive
% materials consisting of Agarose, Gd, CaCO3 and SiO2.

%function mValues = fitPar(optVals,path)
%if nargin ==1
    optVals = [1000,30,-100];
%   path = pwd 
%end
% Read csv file
path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/Fit code';
fParams = csvread([path filesep 'fitParameters.csv']);

% Correct inputs
optVals(1) = 1/optVals(1);
optVals(2) = 1/optVals(2);

% Define concentration search range
gdc = 0:0.05:0.5;
agc = 0:0.5:4;
cac = 0:0.5:5;
sic = 0:0.5:5;

eV =0;
err = [];

for gdi = 1:length(gdc)
   for agi = 1:length(agc)
      for cai = 1:length(cac)
          for sii = 1:length(sic)
            xV = [1,gdc(gdi),agc(agi),cac(cai),sic(sii)];  
            % Calculate error
            xI = fParams*xV';
            eV = eV +1;
            err(eV,:) = abs(xI-optVals');
            xVi(eV,:) = xV;
          end
      end
   end
end