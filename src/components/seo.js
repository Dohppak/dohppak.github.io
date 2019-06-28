import React from 'react';
import Helmet from 'react-helmet';

function Metatags(props) {
    return (
        <Helmet
            title={props.title}
            meta={[
                { name: 'title', content: props.title },

                { name: 'tags', content: props.tags },
                {
                    property: 'og:title',
                    content: props.title,
                },
                {
                    property: 'og:url',
                    content: props.pathname ? props.url + props.pathname : props.url,
                },
                {
                    property: 'og.tags',
                    content: props.tags,
                },

                {
                    property: 'og:image:width',
                    content: '1200',
                },

                {
                    property: 'og:image:height',
                    content: '630',
                },
                {
                    property: 'og:locale',
                    content: 'en',
                },
                { name: 'twitter:card', content: 'summary_large_image' },

                { name: 'twitter:title', content: props.title },

                {
                    name: 'twitter.tags',
                    content: props.tags,
                },
                { property: 'og:type', content: 'website' },
                { name: 'robots', content: 'index, follow' },

                { name: 'twitter:creator', content: '@seungheondoh' },
                { property: 'og:site_name', content: 'https://seungheondoh.netlify.com/' }
            ]}
        >
            <html lang="en" />
        </Helmet>
    )
}

export default Metatags;