#include <mlpack.hpp>
#include "catch_all.hpp"

using namespace std;

int main(int argc, char** argv)
{
    Catch::Session session;
    const int returnCode = session.applyCommandLine(argc, argv);    
    if (returnCode != 0)
        return returnCode;

    cout << "mlpack version: " << mlpack::util::GetVersion() << std::endl;
    cout << "armadillo version: " << arma::arma_version::as_string()
         << std::endl;    

    return session.run();
}
