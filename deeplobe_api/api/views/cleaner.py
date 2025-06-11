import threading

from rest_framework.views import APIView
from rest_framework.response import Response

from deeplobe_api.api.views.function_calls.cleaner import clean_aimodel_local_folder


class DataFolderCleanerEndpoint(APIView):

    """
    API endpoint that allows to clean local data of aimodels.

    * Requires JWT authentication.
    * This endpoint will allows only GET methods.
    """

    def get(self, request):

        """
        clean local data of aimodels.
        """

        cleaner = threading.Thread(target=clean_aimodel_local_folder, args=[True])
        cleaner.start()
        return Response({"message": "data folder cleanup is started."})
