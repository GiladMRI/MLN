function Out=gIsEmpty(in)
Out=zeros(size(in))>1;
for i=1:numel(in)
    Out(i)=isempty(in{i});
end
reshape(Out,size(in));