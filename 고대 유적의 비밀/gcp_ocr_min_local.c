#include <stdio.h>
#include <string.h>

// Source: gcp_vision_ocr_4x_lines_local.txt
// Original OCR line (kept as comment to preserve provenance):
// #include < stdio.h > // char * c_s = " S " ; printf ( c_s ) ; char * c_n = " N " ; printf ( c_n ) ; char * c_v = " V " ; printf ( c_v ) ; r = lambda x : input ( " TURNAROUND " ) ;

static void p(const char *x) {
    printf("%s", x);
}

int main(void) {
    char input_buf[256];
    const char *q = "HALT";   // OCR: q="TL"; q+="AH"; q=q[::-1]
    const char *a = "WRONG";  // OCR: a="G"; a+="NO"; b="R"; b+="W"; a+=b; a=a[::-1]

    // The SNV tokens are within a C comment in OCR, so do not print them.
    p("E");
    p("Q");
    p("X");
    p("CURSED");

    if (fgets(input_buf, sizeof(input_buf), stdin)) {
        size_t len = 0;
        while (input_buf[len] != '\0' && input_buf[len] != '\n' && input_buf[len] != '\r') {
            len++;
        }
        input_buf[len] = '\0';
        if (input_buf[0] != '\0' && strcmp(input_buf, q) != 0) {
            p(a);
        }
    }
    return 0;
}

/*
The remaining OCR text contains mixed Python-like tokens and OCR noise that
are not valid C. It is preserved here for reference only.

a = " G " ; i = input ; a + = " NO " ; q = " TL " ; b = " R " ; b + = " W " ; a + = b ; q + = " AH " ; q = q [ :: - 1 ] ; a = a [ :: - 1 ] ; p = lambda x : print ( x , end = b [ : 0 ] ) or ( exit ( ) ) ;
void ( p ) ( char * x ) { printf ( x ) ; } ; int ( main ) ( ) { char * c_e = " E " ; p ( c_e ) ; char * c_q = " Q " ; p ( c_q ) ; char * c_x = " X " ; p ( c_x ) ; p ( " CURSED " ) ; char * c_u = " " " ;
if ( i ( ) ! = q ) : p ( a ) ; 83j ;
eval ; #char c [ 8888 ] ; c ;
8 // 8 ; eval ; hex ; i = input ; r = " A " ; i = input ; 8 ; i = input ; p = print ; 8 ; 8 // 8 ; eval ;
eval ; 8 // 8 ; False ; r = " L " ; 888 ; r = " T " ; r + = " Q " ; 88 ; ( [ ] ) ; eval ;
eval ; ( [ ] ) ; i = input ; 8 // 8 ; eval ;
...
*/
