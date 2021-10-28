#include <iostream>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <limits>

int main() {
    std::cout.precision(10);
    std::cout << "Programm für Hartree-Fock-Rechnungen an atomaren Systemen mit bis zu 2 Elektronen." << std::endl;

    int z; // Ordnungszahl des Kerns
    int n_alpha, n_beta; // Anzahl der alpha/beta Spinorbitale
    int n_gto; // Anzahl der s-artigen Gauß-Funktionen

    std::cout << std::endl << "Ordnungszahl des Atomkerns:" << std::endl;
    std::cin >> z;

    std::cout << "Anzahl der alpha-Spinorbitale:" << std::endl;
    std::cin >> n_alpha;

    std::cout << "Anzahl der beta-Spinorbitale:" << std::endl;
    std::cin >> n_beta;

    std::cout << "Anzahl der s-artigen Gauß-Funktionen:" << std::endl;
    std::cin >> n_gto;

    double zeta_k[n_gto];
    for (int i=0;i<n_gto;i++) {
        std::cout << "Exponent der " << i+1 << "-ten Gauzßfunktion:" << std::endl;
        std::cin >> zeta_k[i];
    }

    // Ausgabe der vom User eingegebenen Werte
    std::cout << std::endl << "Eingegebene Werte:" << std::endl;
    std::cout << "Z: " << z << std::endl
            << "n_a: " << n_alpha << std::endl
            << "n_b: " << n_beta << std::endl
            << "N: " << n_gto << std::endl;

    std::cout << "Exp:" << std::endl;
    for(int i=0;i<n_gto;i++) {std::cout << "zeta_" << i << ": " << zeta_k[i] << std::endl;}

    //Berechnung der Normierungskonstanten norm_k
    double norm_k[n_gto];
    for (int i=0;i<n_gto;i++) {
        norm_k[i] = pow((2.0*zeta_k[i]/M_PI), 3.0/4.0);
        //std::cout << i << ": " << norm_k[i] << std::endl;
    }

    //std::cout << "N_k:" << std::endl;
    //for(int i=0;i<n_gto;i++) {std::cout << "N_" << i << ": " << norm_k[i] << std::endl;}

    // Initialisieren der Matrizen S, V, T und H
    Eigen::MatrixXd s(n_gto, n_gto);
    Eigen::MatrixXd v(n_gto, n_gto);
    Eigen::MatrixXd t(n_gto, n_gto);
    Eigen::MatrixXd h(n_gto, n_gto);

    // Berechnung der Überlappungsmatrix s
    for (int i=0;i<n_gto;i++) {
        for (int j=0;j<=i;j++) { // Diagonal symmetrisch
            double temp = norm_k[i] * norm_k[j] * pow(M_PI/(zeta_k[i]+zeta_k[j]), 3.0/2.0);
            s(i, j) = temp;
            s(j, i) = temp;
        }
    }


    // Berechnung der Elektron-Kern-Anziehungsmatrix v
    for (int i=0;i<n_gto;i++) {
        for (int j=0;j<=i;j++) { // Diagonal symmetrisch
            double temp = -norm_k[i] * norm_k[j] * 2.0 * M_PI * z / (zeta_k[i]+zeta_k[j]);
            v(i, j) = temp;
            v(j, i) = temp;
        }
    }


    // Berechnung der kinetischen Energie t
    for (int i=0;i<n_gto;i++) {
        for (int j=0;j<=i;j++) { // Diagonal symmetrisch
            double temp = norm_k[i] * norm_k[j] * 3.0 * pow(M_PI, 3.0/2.0)
                * zeta_k[i]*zeta_k[j] / pow((zeta_k[i]+zeta_k[j]), 5.0/2.0);
            t(i, j) = temp;
            t(j, i) = temp;
        }
    }


    // Berechnung der Einelektronen-Hamilton-Matrix H = V + T
    h = v + t;

    
    /*
    std::cout << "\n\nMatrizen S, V, T, H:\nÜberlappung S:\n" << s
        << "\n\nElektron-Kern-Anziehung V:\n" << v
        << "\n\nKinetische Energie T:\n" << t
        << "\n\nHamiltonian H:\n\n\n" << h << std::endl;
    //*/

    // Diagonalisierung von S
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_s(s);
    Eigen::MatrixXd c = eigensolver_s.eigenvectors();


    // Berechnung von sigma und U
    Eigen::MatrixXd sigma = c.transpose() * s * c;
    Eigen::VectorXd diag = sigma.diagonal();
    Eigen::VectorXd temp(n_gto);
    for (int i=0;i<n_gto;i++) {
        temp(i) = 1.0 / sqrt(diag(i));
    }
    Eigen::MatrixXd u = c * temp.asDiagonal();

    //std::cout << "\n\nSigma:\n\n" << sigma << "\n\n";
    //std::cout << "\n\nC:\n\n" << c << "\n\n";

    //std::cout << "\n\nU:\n\n" << u << "\n\n";

    
    // Transformation von H, Diagonalisierung von H_tilde und Berechnung der Energien
    Eigen::MatrixXd h_tilde = u.transpose() * h * u;

    //std::cout << "\n\n============ H_tilde ===========\n\n" << h_tilde << "\n\n";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_h_tilde(h_tilde);
    Eigen::MatrixXd c_tilde = eigensolver_h_tilde.eigenvectors();

    //std::cout << "\n\nC_Tilde:\n" << c_tilde << "\n";

    Eigen::MatrixXd epsilon = c_tilde.transpose() * h_tilde * c_tilde;
    Eigen::MatrixXd c_null = u * c_tilde;

    //std::cout << "\n\nOrbitalenergien:\n" << epsilon << "\n\n";

    // Berechnung der Zweielektronenintegrale (kl|mn)
    double zwei_elec_int[n_gto][n_gto][n_gto][n_gto];
    for (int k=0;k<n_gto;k++) {
        for (int l=0;l<n_gto;l++) {
            for (int m=0;m<n_gto;m++) {
                for (int n=0;n<n_gto;n++) {
                    double a = zeta_k[k] + zeta_k[l] + zeta_k[m] + zeta_k[n];
                    double b = (zeta_k[k] + zeta_k[l]) * (zeta_k[m] + zeta_k[n]);
                    zwei_elec_int[k][l][m][n] = norm_k[k] * norm_k[l] * norm_k[m] * norm_k[n]
                        * 2 * pow(M_PI, 5.0/2.0) / (sqrt(a) * b);
                }
            }
        }
    }


    // Initialisierung von C_alpha und C_beta
    Eigen::MatrixXd c_alpha = c_null;
    Eigen::MatrixXd c_beta = c_null;

    
    // SCF-Iteration Schleife
    double conv = 0.0000001;
    double diff_energie = 100*conv;
    double energie_alt = epsilon.trace();
    double e_uhf;
    int iter = 0;
    while (diff_energie > conv) {
        iter++;
        std::cout << "\n\n================ " << "Iteration " << iter << " ================\n";

        // Berechnung der Dichtematrizen D_alpha und D_beta
        Eigen::MatrixXd d_alpha(n_gto, n_gto);
        d_alpha = Eigen::MatrixXd::Zero(n_gto, n_gto);
        for (int k=0;k<n_gto;k++) {
            for (int l=0;l<n_gto;l++) {
                for (int i=0;i<n_alpha;i++) {
                    d_alpha(k, l) = c_alpha(k, i) * c_alpha(l, i);
                }
            }
        }

        Eigen::MatrixXd d_beta(n_gto, n_gto);
        d_beta = Eigen::MatrixXd::Zero(n_gto, n_gto);
        for (int k=0;k<n_gto;k++) {
            for (int l=0;l<n_gto;l++) {
                for (int i=0;i<n_beta;i++) {
                    d_beta(k, l) = c_beta(k, i) * c_beta(l, i);
                }
            }
        }

        //std::cout << "\n\nDichtematrizen der " << iter << ". Iteration:\nD_alpha:\n" << d_alpha << "\n\nD_beta:\n" << d_beta << "\n\n";

        // Berechnung der Fock-Matrizen F-alpha und F-beta
        Eigen::MatrixXd f_alpha(n_gto, n_gto);
        for (int k=0;k<n_gto;k++) {
            for (int l=0;l<n_gto;l++) {
                f_alpha(k, l) = h(k, l);
                for (int m=0;m<n_gto;m++) {
                    for (int n=0;n<n_gto;n++) {
                        f_alpha(k, l) += (d_alpha(m, n) + d_beta(m, n)) * zwei_elec_int[k][l][m][n]
                                - d_alpha(m, n) * zwei_elec_int[k][m][l][n];
                    }
                }
            }
        }

        Eigen::MatrixXd f_beta(n_gto, n_gto);
        for (int k=0;k<n_gto;k++) {
            for (int l=0;l<n_gto;l++) {
                f_beta(k, l) = h(k, l);
                for (int m=0;m<n_gto;m++) {
                    for (int n=0;n<n_gto;n++) {
                        f_beta(k, l) += (d_alpha(m, n) + d_beta(m, n)) * zwei_elec_int[k][l][m][n]
                                - d_beta(m, n) * zwei_elec_int[k][m][l][n];
                    }
                }
            }
        }

        // Berechnung der Energie der aktuellen Iteration
        e_uhf = 0.5 * (((f_alpha + h) * d_alpha) + ((f_beta + h) * d_beta)).trace();

        diff_energie = energie_alt - e_uhf;
        energie_alt = e_uhf;

        std::cout << "E = " << e_uhf  << "     dE = " << diff_energie << std::endl;

        // Transformation in die orthonormale Basis und anschließende Diagonalisierung
        Eigen::MatrixXd f_alpha_tilde = c_alpha.transpose() * f_alpha * c_alpha;
        Eigen::MatrixXd f_beta_tilde = c_beta.transpose() * f_beta * c_beta;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_f_alpha_tilde(f_alpha_tilde);
        Eigen::MatrixXd v_tilde = eigensolver_f_alpha_tilde.eigenvectors();
        Eigen::MatrixXd epsilon_alpha = v_tilde.transpose() * f_alpha_tilde * v_tilde;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_f_beta_tilde(f_beta_tilde);
        Eigen::MatrixXd w_tilde = eigensolver_f_beta_tilde.eigenvectors();
        Eigen::MatrixXd epsilon_beta = w_tilde.transpose() * f_beta_tilde * w_tilde;

        //std::cout << "\n\nEpsilon alpha und beta:\n" << epsilon_alpha << "\n\n" << epsilon_beta << "\n\n";

        c_alpha = c_alpha*v_tilde;
        c_beta = c_beta*w_tilde;
    }

    std::cout << "\n\n=== SCF-Procedur fertig nach " << iter << " Iterationen: ===\n"
            //<< "E = "
            << e_uhf << std::endl;
 
    return 0;
}