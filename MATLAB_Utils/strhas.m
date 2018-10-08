% function Out=strhas(s1,s2)
function Out=strhas(s1,s2)
if(~iscell(s1))
    s1={s1};
end;
Out=~gIsEmpty(regexp(s1, s2, 'ignorecase', 'match'));