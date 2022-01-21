#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

// kvp是一个结构体，包含两个C风格字符数组指针：key和val，对应键值和值
// 另外包含一个int类型数据used
typedef struct
{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

#endif
