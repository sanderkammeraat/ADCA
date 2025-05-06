

ims = imread("segmentation/Hydrazine 003 cropped.tif");
figure;
imshow(ims)

%% First understand the image
figure;

imhist(ims)


%%
%%


BW = imbinarize(ims);

figure;

imshow(BW)

%%


D = bwdist(BW);



figure; 
imshow(D,[])


%%

L = watershed(D);

Lrgb = label2rgb(L);

figure;

imshow(ims); hold on

imshow(Lrgb)






%%

BWC = imcomplement(BW);

figure;
imshow(BWC)
s


%% Filter on area


BWA = bwareafilt(BW, [200,400]);

figure;
imshow(BWA)


%%







%% Filter on perimeter

[B,L] = bwboundaries(BWA,"noholes");
figure;
imshow(ims); hold on
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1),'LineWidth', 2)
end

%%
sizes = [];
for i=1:length(B)

    sizes(i) = length(B{i});
end

maxperim = 10;
ind = sizes < maxperim;

labels = 1:length(B);
labels = labels(ind);

Bt = B(ind);

boolL = logical(L);



for i=1:length(labels)

    label = labels(i)-1;

    boolL(L==label)=true;
end

Lt = L;
Lt(~boolL)=0;
BWT = BWA;
BWT(~boolL)=0;
%%

figure;
rgb = label2rgb(Lt ,'jet',[.5 .5 .5]);
imshow(ims); hold on
im2 = imshow(rgb)
im2.AlphaData=0.4