(begin 
    (define (map fn ll) 
        (if (null? ll) 
            ll
            (cons (fn (car ll)) (map fn (cdr ll)))
        )
    )

    (define (filter fn ll)
        (if (null? ll)
            ll  
            (if (fn (car ll))
                (cons (car ll) (filter fn (cdr ll))) 
                (filter fn (cdr ll))
            ) 
        )
    )


    (define (reduce fn ll init)
        (if (null? ll)
            init 
            (reduce fn (cdr ll) (fn init (car ll))) 
        ) 
    )
)