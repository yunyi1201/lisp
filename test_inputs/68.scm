(if (list? 7) 1 0)
(if (list? #t) 1 0)
(if (list? #f) 1 0)
(if (list? ()) 1 0)
(if (list? (list 1 2 3)) 1 0)
(if (list? (append)) 1 0)
(if (list? (car (list 1 2 3))) 1 0)
(if (list? (cons 1 2)) 1 0)
(if (list? (cons 1 (list 2))) 1 0)
(if (list? (cons 1 ())) 1 0)
(if (list? (cons (list 1) (list 2))) 1 0)
(if (list? (car (cons (list 1) (list 2)))) 1 0)
(if (list? (cons 1 (cons 2 (cons 3 (cons 4 (cons (cons 1 (cons 2 (cons 3 ()))) (cons 6 (cons 7 (cons 8 ()))))))))) 1 0)
(if (list? (cons 1 (cons 2 (cons 3 (cons 4 (cons (cons 1 (cons 2 (cons 3 ()))) (cons 6 (cons 7 (cons 8 9))))))))) 1 0)
(if (list?) 1 0)
(if (list? (1 2 3)) 1 0)
(if (list? x) 1 0)
(if (list? 1 2 3) 1 0)
(if (list? (lambda () ())) 1 0)
(if (list? ((lambda () ()))) 1 0)
